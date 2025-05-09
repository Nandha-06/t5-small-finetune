# T5 Fine-tuning Project

## [🔗 Streamlit Demo](https://t5-summarizer-demo.streamlit.app/)

## Project Summary and Objective
This project aims to fine-tune the T5 language model for text summarization . The primary objective is to create a model that can generate concise, accurate summaries of longer text passages while maintaining the context and key information from the original content.

## Dataset Description
The model is fine-tuned on the XSum dataset, which contains professionally written single-sentence summaries of news articles. The dataset consists of:
- Training set: Limited to 20,000 examples for memory efficiency
- Validation set: Limited to 1,000 examples
- Test set: Used for final evaluation

Each example in the dataset contains a full news article and its corresponding human-written summary.

## Model Architecture
- Base model: T5-small (default), with options for T5-base or T5-large
- Task-specific adaptation: Sequence-to-sequence learning for text summarization
- Training hyperparameters:
  - Batch size: 4 (with gradient accumulation steps: 4)
  - Learning rate: 2e-5
  - Training epochs: 1
  - Optimizer: AdamW with weight decay
  - Mixed precision training (FP16) when GPU is available
  - Gradient checkpointing for memory optimization

## Training Logs and Snapshots
The training process includes:
- Checkpoints saved every 200 steps
- Maximum of 5 most recent checkpoints retained
- Evaluation metrics calculated on validation set:
  - ROUGE-1: 22.12
  - ROUGE-2: 4.88
  - ROUGE-L: 15.35
  - ROUGE-Lsum: 16.98

## Result Summary
The fine-tuned model demonstrates its ability to generate coherent and relevant summaries as shown in the evaluation results. While the ROUGE scores indicate room for improvement, sample predictions show the model can capture the key points from longer texts.

Example prediction:
- Original: "The pancreas can be triggered to regenerate itself through a type of fasting diet, say US researchers."
- Generated: "A study by the University of Southern California has found that people with type 1 and type 2 diabetes should not rush off and crash diet if it was done without medical guidance. a diet that regenerated a special type of cell in the pancreas regenerated a hormone called a beta cell."

## Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the Dataset

```bash
python src/preprocess.py
```

### 3. Train the Model

```bash
python src/train.py
```

### 4. Evaluate or Generate Summaries

```bash
python src/model_evaluation.py
```

## Customization

Each script has configurable parameters at the top that can be modified:
- In `preprocess.py`: Data directories, tokenization parameters
- In `train.py`: Model type, batch size, learning rate, training epochs
- In `model_evaluation.py`: Generation parameters, beam size

## Notes

- The default model is T5-small, which trains quickly but with modest performance
- For better results, consider using T5-base or T5-large (requires more GPU memory)
- Training time varies based on hardware configuration 
