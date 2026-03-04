# Legal Text Classifier

Fine-tuning DistilBERT on legal text classification using the ECtHR dataset from LexGLUE.

## Project Structure
```

legal-text-classifier/
├── src/
│   ├── train.py          # Fine-tuning logic
│   ├── evaluate.py       # Evaluation metrics
│   └── inference.py      # Run predictions on new text
├── notebooks/
│   └── exploration.ipynb # EDA and dataset exploration
├── models/               # Saved model weights (gitignored)
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## Setup

### Prerequisites
- Python 3.9+
- Git

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/MayuriKawale/legal-text-classifier.git
cd legal-text-classifier
```

**2. Create and activate virtual environment**
```bash
python -m venv legalbert-env
source legalbert-env/Scripts/activate  # Windows Git Bash
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage
Coming soon.

## Dataset
This project uses the ECtHR (European Court of Human Rights) dataset
from the LexGLUE benchmark.
- Source: https://huggingface.co/datasets/coastalcph/lex_glue
- Task: Article violation classification (ecthr_a)

## Model
- Base model: DistilBERT (distilbert-base-uncased)
- Task: Multi-label legal text classification