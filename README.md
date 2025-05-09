# Alpaca Finetune

This project contains code for fine-tuning the Alpaca language model.

## Project Structure

```
.
├── data/               # Data directory for training and evaluation
├── src/               # Source code directory
├── preprocess.py      # Data preprocessing script
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── notebooks/        # Jupyter notebooks for analysis
└── requirements.txt  # Python dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess the data:
```bash
python preprocess.py
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## License

MIT 