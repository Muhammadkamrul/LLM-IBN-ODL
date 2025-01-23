import json
import argparse
from langchain_chroma import Chroma
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_community.embeddings import OllamaEmbeddings
from ollama import Client
from openai import OpenAI
from formal_specification.prompts_test import SYSTEM_PROMPT
import re
import spacy
import time
import matplotlib.pyplot as plt

def draw_bar_time(avg_processing_times):
    # Extract model names and average times
    models = list(avg_processing_times.keys())
    avg_times = list(avg_processing_times.values())

    # Generate bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, avg_times, color='skyblue')

    # Add value labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontsize=8)

    # Customize plot
    plt.title("Comparison of Average Processing Time for Conflict Detection", fontsize=14)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Average Processing Time (s)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("average_processing_times_conflict_natLang.png", bbox_inches='tight')
    plt.show()

def filter_city_subnet(input_intent):
    # Corrected regex pattern for subnets
    subnet_pattern = r'\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b'

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input_intent)

    # Extract city names
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # Geopolitical entities
    # Extract subnets with regex
    subnets = re.findall(subnet_pattern, input_intent)

    return cities, subnets

my_models = ["qwen"
]

my_models2 = [
"codellama:34b",
"qwq",
"huihui_ai/qwq-fusion",
"huihui_ai/qwq-abliterated",
"marco-o1",
"phi3",
"command-r",
"codellama",
"codegemma",
"llama2",
"llama3",
"llama3.1",
"llama3.2",
"qwen",
"qwen2",
"qwen2.5",
"gemma2:27b",
"zephyr",
"yi",
"starcoder", "codestral",
"openchat",
"mistral",
"mistral-nemo",
"tinyllama",
"deepseek-coder",
"starcoder2",
"dolphin-mistral",
"wizardlm2",
"phi",
"orca-mini", "llava-llama3"
]

current_model = "llama2"

# Load datasets from local files
def load_local_datasets(train_file, test_file):
    with open(train_file, "r") as f:
        trainset = json.load(f)
    with open(test_file, "r") as f:
        testset = json.load(f)
    return trainset, testset

# Save updated datasets to local files
def save_local_datasets(trainset, testset, train_file, test_file):
    with open(train_file, "w") as f:
        json.dump(trainset, f, indent=4)
    with open(test_file, "w") as f:
        json.dump(testset, f, indent=4)

# Add new sample to dataset
def add_sample_to_dataset(dataset, instruction, output):
    dataset.append({
        "instruction": instruction,
        "output": json.dumps(output)
    })

# Conflict detection using LLM
def detect_conflicts_llm(model, client, instruction, trainset):
    # Human language conflict detection

    human_conflict_prompt1 = (
        "Analyze the new instruction I will give you for potential conflicts with existing instruction which I will also give you.\n\n"
        "A conflict occurs if the new instruction contradicts with any existing instructions. To clarify, I will give you 5 examples of conflicts which you should understand:\n\n"
        "1. If instruction 'A' says X can reach Y but instruction 'B' says X can't reach Y, then this is a conflict.\n"
        "2. If instruction 'A' says X can reach Y between time 8 to 12, while instruction 'B' says X can't reach Y between time 8 to 12, then this is a conflict of time.\n"
        "3. If instruction 'A' says X can't reach Y between time 9 to 5, while instruction 'B' says X can reach Y between time 4 to 7, then this is a conflict because there is a common time which is 4 to 5 where both instructions conflict.\n\n"
        "4. If instruction 'A' says Connectivity from roma to 100.0.23.0/24 is established while instruction 'B' says roma can't reach 100.0.23.0/24, then this is a conflict because one says 100.0.23.0/24 is reachable but other says it is not reachable. \n"
        "5. If an instruction says that Traffic from a city can reach 10.0.0.0/24 and 10.0.1.0/24 between time 07:00 and 19:00, while another instruction says Traffic from the same city can't reach 10.0.0.0/24 between 06:00 and 08:00, then this is a conflict as there is contradicting reachability with a common time from 7:00 to 8:00. \n"
        "Use these five examples to learn, analyze, and identify the conflicting instruction. If any conflict is found, provide an explanation of why and where the conflicts are.\n\n"
        "You should display the conflicting instruction(s) if found. Make your response brief.\n\n"
        f"New Instruction: {instruction}\n\n"
        f"Existing Instruction: {trainset}\n\n"
        )

    human_conflict_prompt2 = (
        "Analyze the new instruction for potential conflicts with the existing instruction provided. A conflict occurs if there is a direct contradiction between the two instructions. "
        "Focus only on the cities, subnets, and time ranges explicitly mentioned in both instructions. Ignore unrelated cities or subnets.\n\n"
        "Here are examples of conflicts to guide you:\n"
        "1. If instruction 'A' says X can reach Y but instruction 'B' says X can't reach Y, this is a conflict.\n"
        "2. If instruction 'A' says X can reach Y between time 8:00 to 12:00, while instruction 'B' says X can't reach Y between the same time range, this is a conflict of time.\n"
        "3. If instruction 'A' says X can't reach Y between time 9:00 to 5:00, while instruction 'B' says X can reach Y between time 4:00 to 7:00, this is a conflict due to overlapping time (4:00 to 5:00).\n\n"
        "Use these examples to analyze the provided instructions. Identify conflicts if:\n"
        "- The same city has contradictory reachability instructions for the same subnet.\n"
        "- There is a contradiction regarding time-based reachability for the same city and subnet.\n\n"
        "Provide a clear explanation of the conflict, if any. Otherwise, state there is no conflict.\n\n"
        f"New Instruction: {instruction}\n"
        f"Existing Instruction: {trainset}\n"
        )

    human_conflict_prompt3 = (
        "Analyze the new instruction for potential conflicts with the existing instruction provided. A conflict occurs if there is a direct contradiction between the two instructions. "
        "Focus strictly on the cities and subnets explicitly mentioned in both the new and existing instructions. Ignore other cities or subnets, even if they share the same subnet range (e.g., 100.0.x.0/24).\n\n"
        "Examples of conflicts:\n"
        "1. If instruction 'A' says X can reach Y but instruction 'B' says X can't reach Y, this is a conflict.\n"
        "2. If instruction 'A' says X can reach Y between time 8:00 to 12:00, while instruction 'B' says X can't reach Y between the same time range, this is a conflict of time.\n"
        "\n"
        "Provide a concise explanation of the conflict (if any) in one or two sentences. If no conflict is found, clearly state 'No conflict found.'\n\n"
        f"New Instruction: {instruction}\n"
        f"Existing Instruction: {trainset}\n"
    )

    human_conflict_prompt4 = (
        "You are tasked with detecting conflicts between a new instruction and a set of existing instructions. "
        "A conflict arises if there is a direct or implied contradiction in the connectivity between cities, subnets or time, explicitly mentioned in both instructions. "
        "Analyze the relationships thoroughly and follow these steps:\n"
        "1. Identify the cities, subnets and time(if available) mentioned in the new instruction.\n"
        "2. Compare these with the cities, subnets and time(if available) in each existing instruction.\n"
        "   - Example: - A conflict arises only if the same source (city) is stated to have contradictory connectivity to the same subnet (e.g., City A cannot reach Subnet X vs. City A can reach Subnet X).\n"
        "   - If different sources are mentioned (e.g., City A and City B), there is no conflict, even if they reference the same subnet (e.g., Subnet X).\n"
        "   - If the new instruction says City A cannot reach Subnet X, but an existing instruction says City B can reach Subnet X, this is not a conflict because the sources (City A and City B) are different.\n"
        "3. Provide concise conflict detection output for each existing instruction.\n\n"
        "For each conflict, mention:\n"
        "  - The conflicting instruction.\n"
        "  - The specific part causing the conflict.\n"
        "  - A brief explanation.\n"
        "If no conflicts are found for an instruction, don't print anything about it.\n\n"
        f"New Instruction: {instruction}\n"
        f"Existing Instructions: {trainset}\n"
    )

    
    human_response = client.generate(model=model,
                                     options={
                                         'temperature': 0.6,
                                         'num_ctx': 8192,
                                         'top_p': 0.3,
                                         'num_predict': 1024,
                                         'num_gpu': 99,
                                     },
                                     stream=False,
                                     prompt=human_conflict_prompt4
                                    )
    human_conflict_result = human_response['response']

    return human_conflict_result

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Check conflicts in network policies.")
    parser.add_argument('-e', '--ollama-embedding-url', type=str, default="http://localhost:11434", help="Embedding server URL")
    parser.add_argument('-o', '--ollama-server-url', type=str, default="http://localhost:11435", help="LLM server URL")
    return parser.parse_args()

args = parse_args()

# Load datasets from local files
train_file = "trainset_formal_spec.json"  # Update the actual path
test_file = "testset_formal_spec.json"   # Update the actual path
trainset, testset = load_local_datasets(train_file, test_file)


client = Client(host=args.ollama_server_url, timeout=300)

print("\n\nEnter a new network requirement :")
instruction = input().strip()

avg_processing_times = {}

cities, subnets = filter_city_subnet(instruction)

filtered_trainset = [
        entry for entry in trainset[:10]
        if any(city in entry["instruction"] for city in cities) or
        any(subnet in entry["instruction"] for subnet in subnets)
    ]


for model in my_models:

    counter = 0
    print("\n", counter, ": ", model)
    current_time = time.time()
    # print("\nEnter a new network requirement (or type 'exit' to quit):")
    # instruction = input().strip()
    # if instruction.lower() == 'exit':
    #     break

    # Translate the instruction into JSON
    #translated_policy = translate_instruction(instruction, SYSTEM_PROMPT, client, example_selector)
    #print("\nTranslated Policy:")
    #print(json.dumps(translated_policy, indent=4))

    # Check for conflicts
    #human_conflict_result, json_conflict_result = detect_conflicts_llm(client, instruction, translated_policy, trainset)
    #human_conflict_result = detect_conflicts_llm(model, client, instruction, translated_policy, trainset)

    for dataset_entry in filtered_trainset:
            dataset_instruction = dataset_entry["instruction"]
            human_conflict_result = detect_conflicts_llm(model, client, instruction, dataset_instruction)
            #Display results
            print("\n", "Dataset instruction number ", counter," : ", dataset_entry["instruction"])
            print("\nConflict Detection Results for (Human Language) for model :", model, "\n")
            print(human_conflict_result)
            print("\n==============\n\n")
            counter+=1
    proc_time_s = (time.time() - current_time)/34
    avg_processing_times[model] = proc_time_s

    #print("\nConflict Detection Results (Translated JSON):")
    #print(json_conflict_result)

draw_bar_time(avg_processing_times)

print(avg_processing_times)

#first 10 instructions in trainset
#brussels can't reach 100.0.9.0/24.
"""
1)
100.0.13.0/24 is accessible from roma.
Traffic originating from vienna can reach the subnet 100.0.24.0/24. 


2)
 Connectivity from roma to 100.0.23.0/24 is established.
Connectivity from sofia to 100.0.1.0/24 is established. 


3)
 Connectivity from madrid to 100.0.1.0/24 is established.
To reach 100.0.1.0/24 from madrid, traffic is directed through kiev.
Connectivity from brussels to 100.0.9.0/24 is established.
The route from brussels to 100.0.9.0/24 includes rotterdam.
lyon can reach 100.0.9.0/24.
The path between lyon and 100.0.9.0/24 involves rotterdam.
100.0.1.0/24 is accessible from vienna.
To reach 100.0.1.0/24 from vienna, traffic is directed through kiev.
100.0.4.0/24 is accessible from warsaw.
Traffic from warsaw to 100.0.4.0/24 passes through basel.
luxembourg can reach 100.0.1.0/24.
Routing traffic between luxembourg and 100.0.1.0/24 goes via kiev.
100.0.4.0/24 is accessible from lyon.
Traffic from lyon to 100.0.4.0/24 passes through basel.
The subnet 100.0.9.0/24 is reachable from marseille.
Routing traffic between marseille and 100.0.9.0/24 goes via rotterdam.
Connectivity from praha to 100.0.29.0/24 is established.
The path between praha and 100.0.29.0/24 involves london.
The subnet 100.0.29.0/24 is reachable from sofia.
To reach 100.0.29.0/24 from sofia, traffic is directed through london. 


4)
 frankfurt can reach 100.0.29.0/24.
Traffic from frankfurt to 100.0.29.0/24 passes through london.
The subnet 100.0.9.0/24 is reachable from vienna.
Traffic from vienna to 100.0.9.0/24 passes through rotterdam. 


5)
 Traffic originating from vienna can reach the subnet 100.0.9.0/24.
athens can reach 100.0.3.0/24.
Connectivity from barcelona to 100.0.21.0/24 is established.
Traffic originating from zurich can reach the subnet 100.0.25.0/24.
Traffic originating from barcelona can reach the subnet 100.0.26.0/24.
vienna can reach 100.0.31.0/24.
Connectivity from barcelona to 100.0.32.0/24 is established.
Connectivity from london to 100.0.5.0/24 is established.
100.0.20.0/24 is accessible from luxembourg.
milan can reach 100.0.6.0/24. 


6)
 Traffic originating from warsaw can reach the subnet 100.0.0.0/24. 


7)
 100.0.0.0/24 is accessible from bratislava.
paris can reach 100.0.4.0/24. 


8)
 Traffic originating from milan can reach the subnet 100.0.26.0/24.
The subnet 100.0.24.0/24 is reachable from istanbul. 


9)
 basel can reach 100.0.0.0/24.
Traffic originating from sofia can reach the subnet 100.0.32.0/24.
Connectivity from frankfurt to 100.0.10.0/24 is established.
luxembourg can reach 100.0.24.0/24.
The subnet 100.0.13.0/24 is reachable from sofia.
Connectivity from paris to 100.0.25.0/24 is established.
sofia can reach 100.0.4.0/24.
The subnet 100.0.25.0/24 is reachable from vienna.
The subnet 100.0.23.0/24 is reachable from madrid.
The subnet 100.0.4.0/24 is reachable from geneva. 


10)
 warsaw can reach 100.0.4.0/24.
The path between warsaw and 100.0.4.0/24 involves basel.
Traffic originating from sofia can reach the subnet 100.0.4.0/24.
The route from sofia to 100.0.4.0/24 includes basel. 

"""
