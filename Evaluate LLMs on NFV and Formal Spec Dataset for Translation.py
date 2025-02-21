"""
This code is reused and adapted by me for my non profit research work. The original code is from "https://github.com/tu-nv/ibn_llm/tree/master". Credit goes to the original authors author.

"""

# fix chroma db issue with sqlite3
# https://stackoverflow.com/questions/77004853/chromadb-langchain-with-sentencetransformerembeddingfunction-throwing-sqlite3
import pysqlite3
import sys
import time
import jsondiff
import argparse
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from ollama import Client
from openai import OpenAI
from secret import OPENAI_API_KEY
import json, jsondiff
import numpy as np
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
#import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Process a boolean flag.")
    parser.add_argument('-c', '--continuous-learning', action='store_true', default=False,
                        help='Enable continuous learning (default: False)')
    parser.add_argument('-u', '--use-case', choices=['nfv_conf', 'formal_spec'], required=False, default="nfv_conf",
                        help='Use case ("nfv_conf" or "formal_spec")')
    #parser.add_argument('-e', '--ollama-embedding-url', type=str, default="http://141.223.124.22:11434", help='ollama embedding server url for example selection')
    #parser.add_argument('-o', '--ollama-server-url', type=str, default="http://141.223.124.22:11435", help='ollama server url')
    parser.add_argument('-e', '--ollama-embedding-url', type=str, default="http://localhost:11434", help='ollama embedding server url for example selection')
    parser.add_argument('-o', '--ollama-server-url', type=str, default="http://localhost:11435", help='ollama server url')
    #parser.add_argument('-n', '--num-examples', type=int, nargs='+', default=[7,8], help='List of numbers of running examples. Default to [7,8]')
    parser.add_argument('-n', '--num-examples', type=int, nargs='+', default=[0, 1, 3, 6, 9], help='List of numbers of running examples. Default to [0, 1, 3, 6, 9]')
    return parser.parse_args()

args = parse_args()
print(f"Run with args: {args}")

if args.use_case == "formal_spec":
    from formal_specification.dataset import trainset, testset
    from formal_specification.prompts import SYSTEM_PROMPT
    from formal_specification.utils import compare_result
elif args.use_case == "nfv_conf":
    from nfv_configuration.dataset import trainset, testset
    from nfv_configuration.prompts import SYSTEM_PROMPT
    from nfv_configuration.utils import compare_result
else:
    raise ValueError("Invalid use case")

# ollama_emb = OllamaEmbeddings(
#     model="llama3.1",
#     base_url=args.ollama_embedding_url,
# )

#"deepseek-coder-v2" does not support kshift
my_models = [
"codegemma",
"starcoder2",
"dolphin-mistral",
"wizardlm2",
"phi",
"yi",
"command-r",
"orca-mini", "llava-llama3", "zephyr",
"starcoder", "codestral",
"codellama:34b",
"codellama",
"llama2",
"llama3",
"llama3.1",
"llama3.2",
"qwen",
"qwen2",
"qwen2.5",
"gemma2:27b",
"openchat",
"marco-o1",
"mistral",
"phi3",
"huihui_ai/qwq-abliterated",
"huihui_ai/qwq-fusion",
"qwq",
"mistral-nemo",
"tinyllama",
"deepseek-coder"
]

#new models that can be explored # DeepSeek-R1

# Disable preallocation by manually clearing cache
#torch.cuda.empty_cache()
#torch.cuda.set_per_process_memory_fraction(1.0)  # Limit to 50% of GPU memory

my_models_large = ["llama3.3"]

default_model = "llama2"

ollama_emb = OllamaEmbeddings(
    model=default_model,
    base_url=args.ollama_embedding_url,
)

csv_file = "translate_result_NFV_70b_Model.csv"

# Check if the file already exists
if not os.path.exists(csv_file):
    # Create a new file with the header if it doesn't exist
    pd.DataFrame(columns=["model", "num_examples", "accuracy", "avg_time", "std_time"]).to_csv(csv_file, index=False)


openai_client = OpenAI(api_key=OPENAI_API_KEY)
client = Client(host=args.ollama_server_url , timeout=120)
#"mixtral", "qwen2.5-coder","llava.",
#(does not support) "mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed"
for model in my_models_large:

    for num_examples in args.num_examples:
    #for model in ["formal_spec_ft", "qwen2", "llama3.1", "codellama:34b", "gemma2", "gemma2:27b", "qwen2:72b", "llama3.1:70b"]:
        # too much examples in big model cause timeout
        #if num_examples > 1 and model in ["qwen2:72b", "llama3.1:70b"]:
            #continue
        # this test has same result when add example = false
        if args.continuous_learning and num_examples == 0:
            continue

        # create example selector with one example, then clear the data and add all examples
        # this is a trick to reset data and remove data from continuous learning in previous run
        
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples([trainset[0]], ollama_emb, Chroma, input_keys=["instruction"], k=num_examples, vectorstore_kwargs={"fetch_k": min(num_examples, len(trainset))} )
        example_selector.vectorstore.reset_collection()
        for example in trainset:
            example_selector.add_example(example)

        #print("\n\n=====================================")
        #print(f"Start eval on use case: {args.use_case}, model: {model}, num context examples: {num_examples}, continuous learning: {args.continuous_learning}")
        #sys.stdout.flush()

        correct = 0
        total = 0
        processing_times = []

        for testcase in testset:
            intent = testcase["instruction"]
            expected_output = testcase["output"]
            system_prompt = SYSTEM_PROMPT

            while True:
                try:
                    time.sleep(0.1)
                    current_time = time.time()
                    if num_examples > 0:
                        # we do not use FewShotPromptTemplate due to JSON serialization issues
                        # https://github.com/langchain-ai/langchain/issues/4367
                        examples = example_selector.select_examples({"instruction": intent})
                        example_str = "\n\n\n".join(map(lambda x: "Input: " + x["instruction"] + "\n\nOutput: " + x["output"], examples))
                        system_prompt += example_str + "\n\n\n"

                    if model == "chatgpt":
                        messages = [{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": intent}]
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo", messages=messages
                        )
                        response = response.choices[0].message.content
                    else:
                        response = client.generate(model=model,
                            options={
                                'temperature': 0.6,
                                'num_ctx': 8192,
                                'top_p': 0.3,
                                'num_predict': 1024,
                                'num_gpu': 99,
                                },
                            stream=False,
                            system=system_prompt,
                            prompt=intent,
                            format='json'
                        )
                        actual_output = response['response']
                        #print("\nGot Response\n")

                    proc_time_s = (time.time() - current_time)
                    processing_times.append(proc_time_s)
                    break
                except Exception as e:
                    print("Exception on Input: ", e)
                    sys.stdout.flush()
                    continue

            try:
                expected_output = json.loads(expected_output)
                actual_output = json.loads(actual_output)
                num_correct_translation, total_translation = compare_result(expected_output, actual_output)

                if args.continuous_learning and num_examples > 0 and num_correct_translation < total_translation:
                    example_selector.add_example({"instruction": intent, "output": json.dumps(expected_output)})
                    #print(f"Wrong output, add example!")

                #if num_correct_translation == 0:
                    #print(f"Input: {intent}")
                    #print(f"Expected: {expected_output}")
                    #print(f"Actual: {actual_output}")
                    #print(f"Diff: {jsondiff.diff(expected_output, actual_output)}")
                correct += num_correct_translation
                total += total_translation

                #print("=====================================")
                #print(f"Corrects: {correct}, total: {total}, percent: {(correct/total)*100}, proc time: {proc_time_s}")
                #sys.stdout.flush()
            except Exception as e:
                print("Exception on comparing result: ", e)

        print("=====================================")
        print(f"Finish eval on use case: {args.use_case}, model: {model}, num context examples: {num_examples}, continuous learning: {args.continuous_learning}, testcases: {total}, accuracy: {round((correct/total)*100, 3)}  avg proc time: {round(np.average(processing_times), 1)}, std: \t{round(np.std(processing_times), 1)}")
        #print(f"Finish eval on use case: {args.use_case}, model: {model}, num context examples: {num_examples}, continuous learning: {args.continuous_learning}, testcases: {total}, accuracy: {round((correct/total)*100, 3)}  avg proc time: {round(np.average(processing_times), 1)}, std: \t{round(np.std(processing_times), 1)}, processing time array: {[round(x, 2) for x in processing_times]}")
        
        try:
            # Assume this block runs after successful computation
            my_result = {
                        "model": model,
                        "num_examples": num_examples,
                        "accuracy": round((correct / total) * 100, 2),
                        "avg_time": round(np.average(processing_times), 1),
                        "std_time": round(np.std(processing_times), 1),
                    }

            # Append the new result to the CSV file
            pd.DataFrame([my_result]).to_csv(csv_file, mode='a', header=False, index=False)        

        except Exception as e:
            print(f"Error processing model {model} with {num_examples} examples: {e}")             
        
        sys.stdout.flush()

        if(correct == total):
            break

# Load from CSV
# df_csv = pd.read_csv("llm_results1.csv")
# print("DataFrame loaded from CSV:")
# print(df_csv)
