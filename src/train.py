import nltk
import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from evaluate import load as load_metric  # Import just the load function directly
import os
import torch
import datetime
import sys

# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead!")

# Configuration - modify these variables directly
model_name_or_path = "t5-small"  # Path to pretrained model or model identifier
data_dir = "./processed_data"  # Directory containing the processed datasets
output_dir = "./model_output"  # Directory to save the model checkpoints
checkpoint_dir = "./checkpoints"  # Directory for intermediate checkpoints
save_steps = 200  # Save a checkpoint every 200 steps
save_total_limit = 5  # Keep the 5 most recent checkpoints
batch_size = 4  # Reduced batch size for 8GB VRAM
gradient_accumulation_steps = 4  # Accumulate gradients to simulate larger batch
learning_rate = 2e-5  # Learning rate for training
num_train_epochs = 1  # Number of training epochs
max_train_samples = 20000  # Limit training samples to save memory (set to None to use all)
max_eval_samples = 1000  # Limit evaluation samples
push_to_hub = False  # Whether to push the model to the Hub
hub_model_id = None  # Model identifier for uploading to the Hub
load_dataset_from_hub = False  # Load dataset directly from the hub instead of from disk
disable_evaluation = False  # Set to True to skip evaluation during training if it causes errors
simple_evaluation = True  # Use a simplified evaluation method that doesn't require NLTK sentence tokenization

# Create timestamp for checkpoint naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_name = f"{model_name_or_path.split('/')[-1]}_xsum_{timestamp}"
full_checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name)

# Create checkpoint directory
os.makedirs(full_checkpoint_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {full_checkpoint_dir}")

# Download NLTK data packages - ensuring we have everything needed
print("Downloading required NLTK data packages...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab')
except LookupError:
    print("Could not download punkt_tab. Using simpler sentence splitting.")
    
# Define a fallback sentence tokenizer function that doesn't rely on punkt_tab
def safe_sent_tokenize(text):
    """A safer version of sent_tokenize that won't crash if punkt_tab is missing"""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Fallback to a simpler approach - split on common sentence terminators
        import re
        return re.split(r'(?<=[.!?])\s+', text.strip())

# Load the datasets
if load_dataset_from_hub:
    print("Loading XSum dataset directly from the hub...")
    tokenized_datasets = load_dataset("xsum", trust_remote_code=True)
else:
    print(f"Loading processed datasets from {data_dir}...")
    tokenized_datasets = load_from_disk(data_dir)

# Limit the number of samples for memory efficiency
if max_train_samples is not None:
    print(f"Limiting training samples to {max_train_samples} examples")
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(min(max_train_samples, len(tokenized_datasets["train"]))))
    
if max_eval_samples is not None:
    print(f"Limiting validation samples to {max_eval_samples} examples")
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(min(max_eval_samples, len(tokenized_datasets["validation"]))))

# Load model and tokenizer
print(f"Loading model and tokenizer for {model_name_or_path}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load metric
print("Loading ROUGE metric...")
rouge_metric = load_metric("rouge")

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Replace negative values in predictions as they can cause overflow errors
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    
    # Decode predictions with error handling
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    except OverflowError:
        # If we still get overflow errors, do it one by one with error handling
        decoded_preds = []
        for pred in predictions:
            # Replace any remaining negative values
            pred = np.where(pred < 0, tokenizer.pad_token_id, pred)
            try:
                decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(decoded_pred)
            except Exception as e:
                print(f"Error decoding prediction: {e}")
                # Use an empty string as fallback
                decoded_preds.append("")
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode labels with error handling
    try:
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Error decoding labels: {e}")
        # Do it one by one with error handling
        decoded_labels = []
        for label in labels:
            try:
                decoded_label = tokenizer.decode(label, skip_special_tokens=True)
                decoded_labels.append(decoded_label)
            except Exception:
                decoded_labels.append("")
    
    # Prepare text for ROUGE evaluation
    if simple_evaluation:
        # Simple evaluation - just use the texts as they are, without sentence tokenization
        formatted_preds = [pred.strip() for pred in decoded_preds]
        formatted_labels = [label.strip() for label in decoded_labels]
    else:
        # Full evaluation with sentence tokenization
        try:
            # Rouge expects a newline after each sentence
            formatted_preds = ["\n".join(safe_sent_tokenize(pred.strip())) for pred in decoded_preds]
            formatted_labels = ["\n".join(safe_sent_tokenize(label.strip())) for label in decoded_labels]
        except Exception as e:
            print(f"Error tokenizing sentences: {e}. Using simple evaluation instead.")
            formatted_preds = [pred.strip() for pred in decoded_preds]
            formatted_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE scores with error handling
    try:
        result = rouge_metric.compute(predictions=formatted_preds, references=formatted_labels, use_stemmer=True, use_aggregator=True)
        # Extract results and convert to percentages
        result = {key: value * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        # Return fallback metrics
        result = {
            "rouge1": 0.0,
            "rouge2": 0.0, 
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "gen_len": 0.0
        }
    
    return {k: round(v, 4) for k, v in result.items()}

# Define training arguments with GPU specific options
model_name = model_name_or_path.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
    output_dir=full_checkpoint_dir,
    eval_strategy="steps" if not disable_evaluation else "no",  # Disable evaluation if requested
    eval_steps=save_steps if not disable_evaluation else None,  # Only used if evaluation is enabled
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True if not disable_evaluation else False,  # Only needed with evaluation
    generation_max_length=128,
    generation_num_beams=4,
    fp16=torch.cuda.is_available(),  # Use mixed precision only if GPU is available
    push_to_hub=push_to_hub,
    hub_model_id=hub_model_id,
    # Memory optimization settings
    gradient_checkpointing=True,  # Trade compute for memory
    optim="adamw_torch",  # Use memory-efficient optimizer
    no_cuda=False,  # Make sure CUDA is enabled
    load_best_model_at_end=True if not disable_evaluation else False,  # Only possible with evaluation
    metric_for_best_model="rouge1" if not disable_evaluation else None,  # Only used if evaluation is enabled
    greater_is_better=True,  # Higher ROUGE score is better
)

# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize trainer with GPU settings
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=None if disable_evaluation else tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics if not disable_evaluation else None,
)

# Pre-training GPU check
if device.type == "cuda":
    print("\nVerifying GPU availability before training...")
    # Create a small test tensor on GPU
    test_tensor = torch.ones(1, 1).to(device)
    print(f"Test tensor device: {test_tensor.device}")
    print("GPU is properly configured and ready for training!\n")

# Train the model
print("Starting training...")
print(f"Checkpoints will be saved every {save_steps} steps")
trainer.train()

# Save the final model
final_output_dir = os.path.join(output_dir, f"{model_name}-finetuned-xsum")
print(f"Saving final model to {final_output_dir}...")
os.makedirs(final_output_dir, exist_ok=True)
trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

# Write training summary
with open(os.path.join(final_output_dir, "training_summary.txt"), "w") as f:
    f.write(f"Model: {model_name_or_path}\n")
    f.write(f"Training completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training epochs: {num_train_epochs}\n")
    f.write(f"Batch size: {batch_size} (with gradient accumulation: {batch_size * gradient_accumulation_steps})\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Checkpoints saved to: {full_checkpoint_dir}\n")
    f.write(f"Final model saved to: {final_output_dir}\n")

print(f"All checkpoints saved to: {full_checkpoint_dir}")
print(f"Final model saved to: {final_output_dir}")
print("Training completed!")
