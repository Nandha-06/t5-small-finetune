import nltk
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
import torch
import os
import time

# Configuration - modify these variables directly
model_name_or_path = "./model_output/t5-small-finetuned-xsum"  # Path to your fine-tuned model
data_dir = None  # Directory containing processed datasets (None to download from hub)
max_input_length = 1024  # Maximum input length for the tokenizer
max_target_length = 128  # Maximum target length for the tokenizer
batch_size = 2  # Small batch size for 8GB VRAM
num_beams = 4  # Number of beams for beam search

# Explicitly check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead!")

max_test_samples = 500  # Limit number of test samples to evaluate (None for all)
sample_text = None  # Set this to a string to generate a summary for specific text
# sample_text = """Your long text here for summarization..."""  # Uncomment and modify to test
output_file = "evaluation_results.txt"  # File to save evaluation results

# Memory optimization settings
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32  # Use half precision on GPU
low_cpu_mem_usage = True  # Optimize CPU memory usage

# Download punkt for nltk
nltk.download("punkt", quiet=True)

# Load model and tokenizer with memory optimizations
print(f"Loading model and tokenizer from {model_name_or_path}...")

# Add a retry mechanism for model loading
max_retries = 3
for attempt in range(max_retries):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map="auto" if device.type == "cuda" else None
        )
        
        # Explicitly move to GPU and verify
        if device.type == "cuda":
            model = model.cuda()  # Force model to GPU
            print("Model successfully loaded to GPU")
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {str(e)}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Failed to load model after multiple attempts")
            raise

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Verify GPU usage
if device.type == "cuda":
    print("\nVerifying GPU availability...")
    # Create a small test tensor and run through model
    test_input = tokenizer("summarize: This is a test", return_tensors="pt").to(device)
    with torch.no_grad():
        test_output = model.generate(test_input.input_ids, max_length=20)
    print(f"Generated test output is on device: {test_output.device}")
    print("GPU is properly configured and model is using it!\n")
    # Clean up
    del test_input, test_output
    torch.cuda.empty_cache()

# Set prefix for T5 models
prefix = "summarize: " if any(t5_variant in model_name_or_path for t5_variant in 
                           ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]) else ""

# Check if we're doing single text evaluation or test set evaluation
if sample_text:
    print("Generating summary for the provided sample text...")
    
    # Tokenize input
    inputs = tokenizer(prefix + sample_text, return_tensors="pt", max_length=max_input_length, 
                      truncation=True).to(device)
    
    # Generate summary
    with torch.cuda.amp.autocast(enabled=device.type=="cuda"):  # Use automatic mixed precision
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=max_target_length,
            min_length=int(max_target_length / 2),
            length_penalty=2.0,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode and print the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("\nGenerated Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    
    # Save to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Original Text:\n")
        f.write(sample_text + "\n\n")
        f.write("Generated Summary:\n")
        f.write(summary)
    
    print(f"Results saved to {output_file}")
    
else:
    # Load test dataset
    if data_dir:
        print(f"Loading processed datasets from {data_dir}...")
        tokenized_datasets = load_from_disk(data_dir)
        test_dataset = tokenized_datasets["test"]
    else:
        print("Loading and preprocessing XSum dataset...")
        raw_datasets = load_dataset("xsum", trust_remote_code=True)
        test_dataset = raw_datasets["test"]
    
    # Limit test samples for memory efficiency
    if max_test_samples is not None:
        print(f"Limiting test samples to {max_test_samples}")
        test_dataset = test_dataset.select(range(min(max_test_samples, len(test_dataset))))
    
    # Load ROUGE metric
    rouge_metric = evaluate.load("rouge")
    
    # Generate summaries for test set and evaluate
    print("Generating summaries and computing ROUGE scores...")
    all_preds = []
    all_labels = []
    
    # Process in smaller batches to avoid OOM errors
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        
        # Prepare inputs
        inputs = [prefix + doc for doc in batch["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, 
                               padding=True, return_tensors="pt").to(device)
        
        # Generate summaries with mixed precision
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            with torch.no_grad():  # Disable gradient calculation to save memory
                summary_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_length=max_target_length,
                    min_length=int(max_target_length / 2),
                    length_penalty=2.0,
                    num_beams=num_beams,
                    early_stopping=True
                )
        
        # Move tensors to CPU before decoding to free up GPU memory
        summary_ids = summary_ids.detach().cpu()
        
        # Decode generated summaries
        decoded_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        all_preds.extend(decoded_preds)
        all_labels.extend(batch["summary"])
        
        # Print progress
        if (i // batch_size) % 5 == 0:
            progress = (i + len(batch)) / len(test_dataset) * 100
            print(f"Progress: {progress:.2f}% ({i + len(batch)}/{len(test_dataset)})")
            
        # Clear GPU cache periodically
        if device.type == "cuda" and (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Format predictions and references for ROUGE
    formatted_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in all_preds]
    formatted_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in all_labels]
    
    # Compute ROUGE scores
    result = rouge_metric.compute(predictions=formatted_preds, references=formatted_labels, 
                                use_stemmer=True, use_aggregator=True)
    
    # Convert scores to percentages and round
    result = {key: round(value * 100, 2) for key, value in result.items()}
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, score in result.items():
        print(f"{metric}: {score}")
    
    # Save sample predictions and results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Evaluation Results:\n")
        f.write("-" * 50 + "\n")
        for metric, score in result.items():
            f.write(f"{metric}: {score}\n")
        
        f.write("\nSample Predictions:\n")
        f.write("-" * 50 + "\n")
        for i in range(min(5, len(all_preds))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Original: {all_labels[i]}\n")
            f.write(f"Generated: {all_preds[i]}\n\n")
    
    print(f"Results saved to {output_file}") 