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
    plt.savefig("average_processing_times_conflict_json.png", bbox_inches='tight')
    #plt.show()

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

my_models = [
    "tinyllama", "orca-mini", 
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
"deepseek-coder",
"starcoder2",
"dolphin-mistral",
"wizardlm2",
"phi",
"llava-llama3"
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

    json_conflict_prompt1 = (
        "You are tasked with detecting conflicts between a new policy and a set of existing JSON-formatted policies. "
        "A conflict arises if there is a direct or implied contradiction in the 'reachability', 'unreachability', 'waypoints', 'loadbalancing', or 'time' explicitly mentioned in the JSON structures. "
        "Follow these steps carefully to analyze for conflicts:\n\n"

        "1. **Identify the Keys in the New Policy**:\n"
        "   - Extract the 'reachability', 'unreachability', 'waypoint', 'loadbalancing', and 'time' fields.\n"
        "   - Parse the cities, subnets, paths, and time constraints provided in these fields.\n\n"

        "2. **Compare the New Policy with Each Existing Policy**:\n"
        "   - Look for contradictions in 'reachability' and 'unreachability':\n"
        "     - Example Conflict: {'unreachability': {'cityA': ['subnetX']}} contradicts {'reachability': {'cityA': ['subnetX']}}.\n"
        "   - Look for contradictions in 'waypoints' (e.g., the path specified for a subnet is different):\n"
        "     - Example Conflict: {'waypoint': {'(cityA, subnetX)': ['node1']}} contradicts {'waypoint': {'(cityA, subnetX)': ['node2']}}.\n"
        "   - Look for contradictions in 'loadbalancing' (e.g., the number of paths specified for traffic differs):\n"
        "     - Example Conflict: {'loadbalancing': {'(cityA, subnetX)': 2}} contradicts {'loadbalancing': {'(cityA, subnetX)': 3}}.\n"
        "   - Check for conflicts in 'time' constraints:\n"
        "     - Example Conflict: {'time': {'cityA': {'subnetX': ['08:00-12:00']}}} contradicts {'time': {'cityA': {'subnetX': ['09:00-11:00']}}} due to overlapping but inconsistent time ranges.\n\n"

        "3. **Steps to Detect a Conflict**:\n"
        "   - For each city and subnet in the new policy, compare its 'reachability', 'unreachability', 'waypoints', 'loadbalancing', and 'time' constraints with the corresponding fields in each existing policy.\n"
        "   - If you detect a contradiction, flag it as a conflict and provide the following details:\n"
        "       - The conflicting part (reachability, unreachability, waypoint, loadbalancing, or time).\n"
        "       - The specific JSON keys and values causing the conflict.\n"
        "       - A brief explanation of why this is a conflict.\n"
        "   - If no conflicts are found for a specific policy, skip it (do not print anything about it).\n\n"

        "### Example Conflict Detection\n"
        "New Policy:\n"
        "{'unreachability': {'cityA': ['subnetX']}, 'reachability': {'cityA': ['subnetY']}, 'waypoint': {}, 'loadbalancing': {}, 'time': {}}\n\n"
        "Existing Policy:\n"
        "{'reachability': {'cityA': ['subnetX']}, 'reachability': {'cityA': ['subnetZ']}, 'waypoint': {}, 'loadbalancing': {}, 'time': {}}\n\n"
        "**Conflict**:\n"
        "   - Conflicting Part: Reachability vs. Unreachability\n"
        "   - New Policy: cityA CANNOT reach subnetX (unreachability).\n"
        "   - Existing Policy: cityA CAN reach subnetX (reachability).\n"
        "   - Explanation: The new policy states that cityA cannot reach subnetX, while the existing policy explicitly allows cityA to reach subnetX. This is a direct conflict.\n\n"

        "Now analyze the following policies and detect conflicts:\n\n"
        f"New Policy: {instruction}\n"
        f"Existing Policies: {trainset}\n"
    )


    
    json_response = client.generate(model=model,
                                     options={
                                         'temperature': 0.6,
                                         'num_ctx': 8192,
                                         'top_p': 0.3,
                                         'num_predict': 1024,
                                         'num_gpu': 99,
                                     },
                                     stream=False,
                                     prompt=json_conflict_prompt1
                                    )
    json_conflict_result = json_response['response']

    """
    # JSON conflict detection
    json_conflict_prompt2 = (
        f"Analyze the following new translated policy for potential conflicts with existing policies in the dataset.\n\n"
        f"New Policy:\n{json.dumps(translated_policy, indent=4)}\n\n"
        "Dataset Policies:\n" +
        "\n".join([entry["output"] for entry in trainset]) + "\n\n"
        "Conflict Rules:\n"
        "1. A conflict occurs if there is overlapping or contradictory reachability and unreachability destinations.\n"
        "2. If time is present, it applies to the whole policy, not just reachability or unreachability.\n"
        "3. Waypoint and loadbalancing apply only to reachability.\n"
        "4. There should be no common destinations in both reachability and unreachability.\n"
        "5. If time overlaps between conflicting policies, consider the conflict valid.\n"
        "Analyze and identify the conflicting policy or policies. Provide an explanation."
    )
    

   
    json_response = client.generate(model=current_model,
                                    options={
                                        'temperature': 0.6,
                                        'num_ctx': 8192,
                                        'top_p': 0.3,
                                        'num_predict': 1024,
                                        'num_gpu': 99,
                                    },
                                    stream=False,
                                    prompt=json_conflict_prompt,
                                    format='json')
    json_conflict_result = json_response['response']
    """
    #return human_conflict_result, json_conflict_result
    return json_conflict_result

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Check conflicts in network policies.")
    parser.add_argument('-e', '--ollama-embedding-url', type=str, default="http://localhost:11434", help="Embedding server URL")
    parser.add_argument('-o', '--ollama-server-url', type=str, default="http://localhost:11435", help="LLM server URL")
    return parser.parse_args()

args = parse_args()

# Load datasets from local files
train_file = "modified_trainset.json"  # Update the actual path
test_file = "modified_testset.json"   # Update the actual path
trainset, testset = load_local_datasets(train_file, test_file)

client = Client(host=args.ollama_server_url, timeout=300)

print("\n\nEnter a new network requirement :")
instruction = input().strip()

avg_processing_times = {}

# cities, subnets = filter_city_subnet(instruction)
# print(cities)
# print(subnets)
# print("\n\n")

# filtered_trainset = [
#         entry for entry in trainset[:10]
#         if any(city in entry["output"] for city in cities) or
#         any(subnet in entry["output"] for subnet in subnets)
#     ]


for model in my_models:

    counter = 0
    print("\n", counter, ": ", model)
    current_time = time.time()

    for dataset_entry in trainset[:10]:
            dataset_instruction = dataset_entry["output"]

            # print("\n==============================")
            # print(dataset_instruction)
            # print("================================\n")
            # continue

            json_conflict_result = detect_conflicts_llm(model, client, instruction, dataset_instruction)
            
            #Display results
            print("\n", "Dataset instruction number ", counter," : ", dataset_entry["output"])
            print("\nConflict Detection Results for (JSON) for model :", model, "\n")
            print(json_conflict_result)
            print("\n==============\n\n")
            counter+=1

    proc_time_s = (time.time() - current_time)/10
    avg_processing_times[model] = proc_time_s

draw_bar_time(avg_processing_times)
print(avg_processing_times)

# List of models
# models = [ "codellama:34b","qwq","huihui_ai/qwq-fusion","huihui_ai/qwq-abliterated","marco-o1","phi3","command-r","codellama","codegemma","llama2","llama3","llama3.1",
#"llama3.2","qwen","qwen2","qwen2.5","gemma2:27b","zephyr","yi","starcoder", "codestral","openchat","mistral","mistral-nemo","tinyllama","deepseek-coder","starcoder2",
#"dolphin-mistral","wizardlm2","phi","orca-mini", "llava-llama3"       ]

# Correct conflict detection counts out of 34 inputs
#correct_counts = [  ]

"""
#first 10 instructions in trainset
#brussels can't reach 100.0.9.0/24.
{"reachability": {"sofia": ["100.0.1.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {"roma": ["100.0.23.0/24"]}, "time": null}

start:
{"reachability": {"roma": ["100.0.13.0/24"], "vienna": ["100.0.24.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"roma": ["100.0.23.0/24"], "sofia": ["100.0.1.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 



{"reachability": {"madrid": ["100.0.1.0/24"], "brussels": ["100.0.9.0/24"], "lyon": ["100.0.9.0/24", "100.0.4.0/24"], "vienna": ["100.0.1.0/24"], "warsaw": ["100.0.4.0/24"], "luxembourg": ["100.0.1.0/24"], "marseille": ["100.0.9.0/24"], "praha": ["100.0.29.0/24"], "sofia": ["100.0.29.0/24"]}, "waypoint": {"(madrid,100.0.1.0/24)": ["kiev"], "(brussels,100.0.9.0/24)": ["rotterdam"], "(lyon,100.0.9.0/24)": ["rotterdam"], "(vienna,100.0.1.0/24)": ["kiev"], "(warsaw,100.0.4.0/24)": ["basel"], "(luxembourg,100.0.1.0/24)": ["kiev"], "(lyon,100.0.4.0/24)": ["basel"], "(marseille,100.0.9.0/24)": ["rotterdam"], "(praha,100.0.29.0/24)": ["london"], "(sofia,100.0.29.0/24)": ["london"]}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"frankfurt": ["100.0.29.0/24"], "vienna": ["100.0.9.0/24"]}, "waypoint": {"(frankfurt,100.0.29.0/24)": ["london"], "(vienna,100.0.9.0/24)": ["rotterdam"]}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"vienna": ["100.0.9.0/24", "100.0.31.0/24"], "athens": ["100.0.3.0/24"], "barcelona": ["100.0.21.0/24", "100.0.26.0/24", "100.0.32.0/24"], "zurich": ["100.0.25.0/24"], "london": ["100.0.5.0/24"], "luxembourg": ["100.0.20.0/24"], "milan": ["100.0.6.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"warsaw": ["100.0.0.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"bratislava": ["100.0.0.0/24"], "paris": ["100.0.4.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"milan": ["100.0.26.0/24"], "istanbul": ["100.0.24.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"basel": ["100.0.0.0/24"], "sofia": ["100.0.32.0/24", "100.0.13.0/24", "100.0.4.0/24"], "frankfurt": ["100.0.10.0/24"], "luxembourg": ["100.0.24.0/24"], "paris": ["100.0.25.0/24"], "vienna": ["100.0.25.0/24"], "madrid": ["100.0.23.0/24"], "geneva": ["100.0.4.0/24"]}, "waypoint": {}, "loadbalancing": {}, "unreachability": {}, "time": null} 

{"reachability": {"warsaw": ["100.0.4.0/24"], "sofia": ["100.0.4.0/24"]}, "waypoint": {"(warsaw,100.0.4.0/24)": ["basel"], "(sofia,100.0.4.0/24)": ["basel"]}, "loadbalancing": {}, "unreachability": {}, "time": null} 

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