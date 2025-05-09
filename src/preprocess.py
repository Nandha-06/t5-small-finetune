from datasets import load_dataset
from transformers import AutoTokenizer
import os
import json

# Configuration - modify these variables directly instead of using command line arguments
model_name = "t5-small"  # Model to use (t5-small, t5-base, etc.)
max_input_length = 1024  # Maximum length for article text
max_target_length = 128  # Maximum length for summaries
output_dir = "./processed_data"  # Where to save tokenized data
raw_output_dir = "./raw_data"  # Where to save original text data
raw_samples = 10  # Number of examples to save as readable JSON

# Load dataset
print("Loading dataset...")
raw_datasets = load_dataset("xsum", trust_remote_code=True)

# Save the raw dataset
os.makedirs(raw_output_dir, exist_ok=True)

print(f"Saving raw datasets to {raw_output_dir}...")
raw_datasets.save_to_disk(raw_output_dir)

# Save sample examples in human-readable JSON format
print(f"Saving {raw_samples} sample examples as human-readable JSON...")
for split in raw_datasets.keys():
    samples = raw_datasets[split].select(range(min(raw_samples, len(raw_datasets[split]))))
    samples_list = []
    
    for sample in samples:
        samples_list.append({
            "document": sample["document"],
            "summary": sample["summary"],
            "id": sample["id"]
        })
    
    # Save samples to a readable JSON file
    samples_file = os.path.join(raw_output_dir, f"{split}_samples.json")
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples_list, f, indent=2, ensure_ascii=False)

# Load tokenizer
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set prefix for T5 models
prefix = "summarize: " if model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] else ""

# Define preprocessing function
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process the datasets
print("Preprocessing datasets...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save processed datasets
print(f"Saving processed datasets to {output_dir}...")
tokenized_datasets.save_to_disk(output_dir)

print("Preprocessing completed!") 