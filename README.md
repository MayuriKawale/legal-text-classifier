# Legal Text Classifier

A multi-label text classification system that predicts which articles 
of the European Convention on Human Rights (ECHR) were violated, 
based on case facts from the European Court of Human Rights (ECtHR).

Built by fine-tuning [DistilBERT](https://huggingface.co/distilbert-base-uncased) 
on the [LexGLUE benchmark](https://huggingface.co/datasets/coastalcph/lex_glue) 
(`ecthr_a` task which predicts violated articles from case facts).

## Dataset

- **Source:** [LexGLUE benchmark](https://huggingface.co/datasets/coastalcph/lex_glue) (`ecthr_a` task)
- **Task:** Multi-label classification that predicts which ECHR articles were violated
- **Size:** 11,000 cases (9,000 train / 1,000 validation / 1,000 test)
- **Features:** Case facts (list of paragraphs) and violated article labels

### Label Distribution
| Article | Right Protected | Train Count | % |
|---------|----------------|-------------|---|
| 6 | Right to a fair trial | 4,704 | 52.27% |
| P1-1 | Protection of property | 1,421 | 15.79% |
| 5 | Right to liberty and security | 1,368 | 15.20% |
| 3 | Prohibition of torture | 1,349 | 14.99% |
| 8 | Right to privacy | 710 | 7.89% |
| 2 | Right to life | 505 | 5.61% |
| 10 | Freedom of expression | 291 | 3.23% |
| 14 | Prohibition of discrimination | 141 | 1.57% |
| 11 | Freedom of assembly | 110 | 1.22% |
| 9 | Freedom of thought | 41 | 0.46% |

### Key Characteristics
- **Multi-label** : Each case can violate multiple articles simultaneously
- **Severe class imbalance**: Article 6 appears 114x more than Article 9
- **Long documents**: Average 1,619 words per case, 72.9% exceed DistilBERT's
  512 token limit

## Model

- **Base model:** [DistilBERT](https://huggingface.co/distilbert-base-uncased) 
  (`distilbert-base-uncased`)
- **Task:** Multi-label sequence classification
- **Loss function:** Binary Cross Entropy (BCE) for 10 independent binary 
  decisions, one per article
- **Threshold:** 0.5, probabilities above this are predicted as violated

### Why DistilBERT?
- 40% smaller and 60% faster than BERT while retaining 97% of performance
- Well suited for fine-tuning on domain specific text
- 512 token limit is manageable for most ECtHR case facts

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 512 tokens |
| Optimizer | AdamW |
| Hardware | Google Colab T4 GPU |
| Training time | ~20 minutes |

### Truncation Strategy
72.9% of cases exceed the 512 token limit. Simple truncation was applied, 
keeping only the first 512 tokens. This is acceptable because key facts 
consistently appear in early paragraphs across all article classes, as 
verified during exploratory data analysis.

## Project Structure
```
legal-text-classifier/
├── src/
│   ├── __init__.py        # marks src/ as a Python package
│   ├── train.py           # fine-tuning pipeline
│   ├── evaluate.py        # evaluation metrics
│   └── inference.py       # predict on new legal text
├── notebooks/
│   ├── exploration.ipynb  # exploratory data analysis
│   └── colab_training.ipynb # training on Google Colab
├── tests/
│   ├── test_train.py      # tests for training pipeline
│   ├── test_evaluate.py   # tests for evaluation pipeline
│   └── test_inference.py  # tests for inference pipeline
├── models/                # saved model weights (gitignored)
├── conftest.py            # pytest configuration for imports
├── pytest.ini             # pytest settings
├── main.py                # entry point with argparse
└── requirements.txt       # project dependencies
```

## Setup & Installation

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
# source legalbert-env/bin/activate    # Mac/Linux
```

**3. Install dependencies**
```bash
# pip install -r requirements.txt      # exact dependicies with version
pip install transformers datasets scikit-learn torch evaluate accelerate
```

**4. Run tests to verify installation**
```bash
pytest tests/ -v                       # run all tests at once
# pytest tests/test_train.py -v        # run tests for training
# pytest tests/test_evaluate.py -v     # run tests for evaluation
# pytest tests/test_inference.py -v    # run tests for inference
```

## Usage

### Train
Fine-tune DistilBERT on the ECtHR dataset:
```bash
python main.py --mode train
```

### Debug Mode
Run a single batch to verify the pipeline works:
```bash
python main.py --mode train --debug
```

### Evaluate
Evaluate the fine-tuned model on the test set:
```bash
python main.py --mode evaluate
```

### Inference
Predict violated ECHR articles for a given legal text:
```bash
python main.py --mode inference --text "The applicant was detained for 14 months 
without being brought before a judge. His requests for release were repeatedly 
denied without explanation. The domestic courts failed to provide adequate 
reasoning for continued detention."
```

### Help
```bash
python main.py --help
```

### Note on Input Text
The model requires sufficient context (typically 100+ words) to make confident 
predictions above the 0.5 threshold. Short texts may return no predictions. 
Provide more detailed case facts for better results.

## Results

Model evaluated on 1,000 held-out test cases after 3 epochs of fine-tuning.

### Overall Metrics
| Metric | Score |
|--------|-------|
| Micro F1 | 0.6393 |
| Macro F1 | 0.5054 |
| Weighted F1 | 0.6227 |

### Per Label F1 Scores
| Article | Right Protected | F1 Score |
|---------|----------------|----------|
| P1-1 | Protection of property | 0.7442 |
| 2 | Right to life | 0.7324 |
| 3 | Prohibition of torture | 0.6980 |
| 5 | Right to liberty and security | 0.6854 |
| 6 | Right to a fair trial | 0.6214 |
| 10 | Freedom of expression | 0.5354 |
| 11 | Freedom of assembly | 0.4815 |
| 8 | Right to privacy | 0.4444 |
| 14 | Prohibition of discrimination | 0.1111 |
| 9 | Freedom of thought | 0.0000 |

### Training History
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.2083 | 0.1913 |
| 2 | 0.1401 | 0.1716 |
| 3 | 0.1156 | 0.1727 |

### Analysis
- **Strong performers** (F1 > 0.65) - Articles 2, 3, 5, 6, P1-1 which have 
  sufficient training examples
- **Moderate performers** (F1 0.4-0.65) - Articles 8, 10, 11 with moderate 
  training examples
- **Poor performers** (F1 < 0.15) - Articles 9 and 14 with very few training 
  examples (41 and 141 respectively)
- **Gap between Micro (0.64) and Macro (0.51) F1** confirms the model 
  struggles with rare labels as predicted during EDA
- **No overfitting** - validation loss closely tracks training loss across 
  all 3 epochs


## Known Limitations & Future Work

### Current Limitations

**1. Truncation**
- 72.9% of cases exceed DistilBERT's 512 token limit
- Only the first ~350 words of each case are seen by the model
- Key legal arguments appearing later in documents are lost

**2. Class Imbalance**
- Article 9 (41 training examples) achieves F1 = 0.0000
- Article 14 (141 training examples) achieves F1 = 0.1111
- Model is biased toward predicting common articles

**3. Threshold**
- Fixed threshold of 0.5 may not be optimal for all use cases
- In legal research, missing a violation may be worse than a false alarm
- A lower threshold would improve recall at the cost of precision

**4. Input Length**
- Model requires sufficient context (100+ words) to make confident predictions
- Short text inputs may return no predictions

### Future Improvements

**1. Longformer**
- Replace DistilBERT with Longformer which supports up to 4,096 tokens
- Would capture full document context instead of truncating at 512 tokens
- Expected to significantly improve performance on long cases

**2. Handle Class Imbalance**
- Assign higher penalty to rare label errors using weighted BCE loss
- Oversample rare label examples during training
- Expected to improve F1 scores for Articles 9, 14 and 11

**3. Threshold Tuning**
- Optimize threshold per label using validation set
- Lower threshold for rare labels to improve recall
- Higher threshold for common labels to improve precision

**4. Extended Training**
- Train for more epochs with early stopping
- Experiment with lower learning rates for more careful fine-tuning
- Try larger batch sizes on GPU for more stable gradient updates

**5. Longformer or Hierarchical Models**
- Explore chunking strategies i.e. classify each 512 token chunk separately
  and aggregate predictions across chunks
- Explore hierarchical models that process paragraph by paragraph

## References

### Dataset
- Chalkidis, I., et al. (2022). *LexGLUE: A Benchmark Dataset for Legal 
  Language Understanding in English.*
  [https://arxiv.org/abs/2110.00976](https://arxiv.org/abs/2110.00976)

### Model
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, 
  faster, cheaper and lighter.*
  [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

### Transformer Architecture
- Vaswani, A., et al. (2017). *Attention Is All You Need.*
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### Tools & Libraries
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/)