import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.train import CONFIG

# ----------- Label names and mapping for reporting -----------
LABEL_NAMES = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']

ARTICLE_DESCRIPTIONS = {
    '2': 'Right to life',
    '3': 'Prohibition of torture',
    '5': 'Right to liberty and security',
    '6': 'Right to a fair trial',
    '8': 'Right to respect for private and family life',
    '9': 'Freedom of thought, conscience and religion',
    '10': 'Freedom of expression',
    '11': 'Freedom of assembly and association',
    '14': 'Prohibition of discrimination',
    'P1-1': 'Protection of property'
}

# ------- Load the saved model and tokenizer -------
def load_model_for_inference(config):
    # load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config["model_save_path"])
    # load the model
    model = DistilBertForSequenceClassification.from_pretrained(config["model_save_path"])
    model.eval()  # set the model to evaluation mode
    return model, tokenizer

#------- Preprocessing function for inference -------
def preprocess_for_inference(text, tokenizer, config):
    ''' Tokenize and truncate the input text for inference '''

    # join the list of sentences into a single string if it's a list
    if isinstance(text, list):
        text = " ".join(text)
    
    # tokenize the input text
    encoding = tokenizer(text, 
                         truncation=True, 
                         padding='max_length', 
                         max_length=config["max_length"], 
                         return_tensors='pt')  # return PyTorch tensors
    return encoding

#-------- Predict article labels for a single input text --------
def predict_article_violated(text, model, tokenizer, config):
    ''' Predict the violated articles for a given legal text.
        Args: text (str or list of str): Input legal text
              model: trained DistilBERT model for sequence classification
              tokenizer: DistilBERT tokenizer
              config: configuration dictionary containing model parameters and threshold
        Returns:
        A list of predicted article labels (e.g., ['2', '3']) and their descriptions.
    '''
    if text is None:
        raise ValueError("Input text cannot be None. Please provide a valid legal text for inference.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # preprocess the input text
    encoding = preprocess_for_inference(text, tokenizer, config)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # forward pass through the model
    with torch.no_grad():  # disable gradient calculation for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs.logits).squeeze(0)  # shape: (num_labels,)

    # Debug — print all probabilities regardless of threshold
    print("\nDebug — All article probabilities:")
    for i, (label, prob) in enumerate(zip(LABEL_NAMES, probs)):
        print(f"  Article {label:>4}: {prob.item():.4f}")

    # Apply threshold to get binary predictions
    results = {}
    for i, (label, prob) in enumerate(zip(LABEL_NAMES, probs)):
        if prob.item() >= config["threshold"]:
            results[label] = {
                "probability": round(prob.item(),4),
                "description": ARTICLE_DESCRIPTIONS[label]
            }
    return results

# -------- Format predictions in a readable format --------
def format_predictions(results):
    ''' Print the predicted articles and their descriptions in a human readable format '''
    if not results:
        print("No articles predicted as violated.")
        return
    
    print('='*50)
    print(f'Predicted Article Violations')
    print('='*50)
    for label, info in sorted(results.items()): # convert results dictionary to a list of tuples and sort by label
        bar = '█' * int(info["probability"] * 20)  # create a bar proportional to the probability
        print(f"Article {label:>4} | {bar:<20} | Probability: {info['probability']:.4f}")
        print(f"             {info['description']}")
        print() # add an empty line for better readability between articles


# -------- Main function to run inference with a sample input --------
if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model_for_inference(CONFIG)
    print("Model loaded successfully.")

    # Sample input text for inference (can be replaced with any legal text)
    sample_text = '''
    The applicant was detained without charge for several days and was not allowed to contact a lawyer.
    During detention, the applicant was subjected to harsh interrogation techniques that caused physical and psychological harm.
    The applicant's family was not informed of the detention, and the applicant was held incommunicado for a prolonged period.
    '''

    print("Running inference on the sample input...")
    results = predict_article_violated(sample_text, model, tokenizer, CONFIG)
    format_predictions(results)


