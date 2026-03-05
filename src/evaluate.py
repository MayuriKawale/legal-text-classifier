# imports and setup
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from src.train import ECtHRDataset, CONFIG

# Label names for reporting
LABEL_NAMES = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']

# ------- Load the saved model and tokenizer -------
def load_saved_model(config):
    # load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config["model_save_path"])
    # load the model
    model = DistilBertForSequenceClassification.from_pretrained(config["model_save_path"])

    model.eval()  # set the model to evaluation mode
    return model, tokenizer

# ------- Get predictions -------
def get_predictions(model, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():            # disable gradient calculation for evaluation
        for batch in test_loader:
            # move inputs, attention masks, and labels to the same device as the model
            input_ids = batch["input_ids"].to(device)  
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Convert logits to probabilities 
            probs = torch.sigmoid(outputs.logits)  # apply sigmoid to get probabilities

            # Convert probabilities to binary predictions using a given threshold (here we use 0.5)
            preds = (probs >= config["threshold"]).int()  # convert to binary predictions (0 or 1)

            all_preds.append(preds.cpu().numpy())  # move predictions to CPU and convert to numpy
            all_labels.append(labels.cpu().numpy())  # move labels to CPU and convert to numpy

    # Stack all predictions and labels for all batches into single numpy arrays
    all_preds = np.vstack(all_preds)  # shape: (num_samples, num_labels)
    all_labels = np.vstack(all_labels)  # shape: (num_samples, num_labels

    return all_preds, all_labels


# ------- Evaluate the model -------
def evaluate(model, test_loader, config):
    print("Running evaluation on the test set...")
    all_preds, all_labels = get_predictions(model, test_loader, config)

    # Calculate F1 score for each label and overall
    # Below: zero_division=0 to handle cases like article 9, which may not be predicted at all due to class imbalance, preventing division by zero errors in F1 score calculation
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0) # micro f1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) # macro f1 score
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) # weighted f1 score

    print("=" * 50)
    print('Overall Metrics:')
    print("=" * 50)
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    # Per label F1 scores
    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0) # f1 score for each label

    print("=" * 50)
    print("\nPer Label F1 Scores:")
    print("=" * 50)
    for label, score in zip(LABEL_NAMES, per_label_f1):
        bar = '█' * int(score * 50)  # create a bar proportional to the F1 score
        print(f"Article {label:>4} | {bar:<20} | F1 Score: {score:.4f}") # ': < 20' to left-align the bar and give it a fixed width

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label_f1": dict(zip(LABEL_NAMES, per_label_f1))
    }

# ------- Main function to run evaluation -------
if __name__ == "__main__":
    # load the saved model and tokenizer
    model, tokenizer = load_saved_model(CONFIG)

    # Load the test dataset
    print("Loading test dataset...")
    raw_dataset = load_dataset("lex_glue", "ecthr_a")
    test_dataset = ECtHRDataset(raw_dataset["test"], tokenizer, CONFIG)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Run evaluation
    metrics = evaluate(model, test_loader, CONFIG)