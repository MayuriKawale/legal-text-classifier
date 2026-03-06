# imports
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import os

# --------- Configurations Dictionary ---------
CONFIG = {
    "model_name": "distilbert-base-uncased",   # Pre-trained model name; uncased because it was trained on lowercased text
    "max_length": 512,                         # Maximum sequence length for tokenization
    "batch_size": 8,                           # Batch size for training and evaluation
    "num_epochs": 3,                           # Number of training epochs
    "learning_rate": 2e-5,                     # Learning rate for the optimizer
    "num_labels": 10,                          # Number of classes for classification
    "model_save_path": "models/",              # Directory to save the trained model
    "threshold": 0.5,                          # Threshold for binary classification
}

# --------- Dataset Class ---------------
class ECtHRDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, config):   # Initialize the dataset with Hugging Face dataset, tokenizer, and configuration
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):                                   # Return the total number of examples in the dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):                          # Retrieve the example at the specified index
        example = self.dataset[idx]
        
        # Step 1: Join the text paragraphs into a single string
        text = " ".join(example['text'])

        # Step 2: Tokenize and truncate the text to 512 tokens (pre-built object from Hugging Face for tokenization)
        encoding = self.tokenizer(text, 
                                  max_length=self.config['max_length'],   # Truncate the text to the maximum length defined in the configuration
                                  padding='max_length',                   # Pad the text to the maximum length if it's shorter than that
                                  truncation=True,                        # Enable truncation to ensure that the text does not exceed the maximum length
                                  return_tensors='pt')                    # Return the tokenized output as PyTorch tensors denoted by 'pt'
        
        # Step 3: Convert the labels into multi-hot encoding for multi-label classification
        labels = torch.zeros(self.config['num_labels'])

        for label in example['labels']:
            labels[label] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),            # Remove batch dimension (without squeeze, the shape would be [1, max_length])
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Remove batch dimension (with squeeze(0), the shape becomes [max_length])
            'labels': labels
        }
    
## Returns a dictionary containing the tokenized input IDs, attention mask, and multi-hot encoded labels for the example at the specified index. 
# input_ids: tensor of shape [max_length] 
# attention_mask: tensor of shape [max_length] 
# labels: tensor of shape [num_labels] (multi-hot encoded labels for multi-label classification)

# --------- Loading Tokenizer and Dataset ---------
def load_data(config):
    # Load the DistilBERT tokenizer using the specified model name from the configuration
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    
    # Load the ECtHR dataset using Hugging Face's datasets library
    dataset = load_dataset("lex_glue", "ecthr_a")
    
    # Create three instances of the ECtHRDataset class: one for training, one for validation, and one for testing
    train_dataset = ECtHRDataset(dataset['train'], tokenizer, config)  # Create training dataset
    val_dataset = ECtHRDataset(dataset['validation'], tokenizer, config)  # Create validation dataset
    test_dataset = ECtHRDataset(dataset['test'], tokenizer, config)  # Create testing dataset

    # Create DataLoader objects for each dataset to enable batching and shuffling (not needed for validation and test)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)  # DataLoader for training dataset with shuffling
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)  # DataLoader for validation dataset without shuffling
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)  # DataLoader for testing dataset without shuffling

    return train_loader, val_loader, test_loader  # Return the DataLoader objects for training, validation, and testing datasets

## Returns three DataLoader objects for the training, validation, and testing datasets.
#  Each dataloader contains: input_ids, attention_mask, and multi-hot encoded labels for the respective dataset.
# input_ids: tensor of shape [batch_size, max_length]
# attention_mask: tensor of shape [batch_size, max_length]
# labels: tensor of shape [batch_size, num_labels] (multi-hot encoded labels for multi-label classification)

# --------- Model Setup ---------
def load_model(config):
    # Load the pre-trained DistilBERT model for sequence classification using the specified model name and number of labels from the configuration
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model_name'], 
        num_labels=config['num_labels'],
        problem_type="multi_label_classification"
    )
    return model 

# ---------- Training and Validation ---------
def train(model, train_loader, val_loader, config, debug=False):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')
    model = model.to(device)  # Move the model to the specified device (GPU or CPU)

    # Define the optimizer (AdamW) with the model parameters and learning rate from the configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate']
    )

    # Training loop for the specified number of epochs
    for epoch in range(config['num_epochs']):

        # Training phase
        model.train()  # Set the model to training mode
        total_train_loss = 0.0  # Initialize total loss for the epoch

        for batch_idx, batch in enumerate(train_loader):
            # Check debug mode first before any computation - for 1 batch only
            if debug and batch_idx == 1:
                print("Debug mode: stopping after 1 batch")
                return model

            # Move batch data to the specified device (GPU or CPU)
            input_ids = batch["input_ids"].to(device)  # Move input IDs to the specified device
            attention_mask = batch["attention_mask"].to(device)  # Move attention mask to the specified device
            labels = batch["labels"].to(device)  # Move labels to the specified device

            # Forward pass through the model to get the outputs (logits and loss)
            optimizer.zero_grad()  # Clear the gradients from the previous step
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Backward pass and optimization step
            loss = outputs.loss  # Get the loss from the model outputs
            loss.backward()  # Backpropagate the loss to compute gradients
            optimizer.step()  # Update the model parameters based on the computed gradients

            total_train_loss += loss.item()  # Accumulate the loss for the epoch

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch +1}/{config["num_epochs"]}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / len(train_loader)  # Calculate average training loss for the epoch
        #print(f'Epoch {epoch + 1} completed. Average Training Loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0  # Initialize total validation loss for the epoch

        with torch.no_grad(): # Disable gradient calculation during validation 
            for batch in val_loader:
                # Move batch data to the specified device (GPU or CPU)
                input_ids = batch["input_ids"].to(device)  
                attention_mask = batch["attention_mask"].to(device)  
                labels = batch["labels"].to(device)

                # Forward pass through the model to get the outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item() # Accumulate the validation loss

        avg_val_loss = total_val_loss / len(val_loader)  # Calculate average validation loss for the epoch

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} completed.")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}\n")

    return model  # Return the trained model after all epochs are completed


# ------------ Save Model ---------
def save_model(model, tokenizer, config):
    os.makedirs(config["model_save_path"], exist_ok=True)  # Create the directory to save the model if it doesn't exist
    model.save_pretrained(config["model_save_path"])  # Save the model's weights and configuration to the specified directory
    tokenizer.save_pretrained(config["model_save_path"])  # Save the tokenizer's configuration and vocabulary to the specified directory
    print(f'Model and tokenizer saved to {config["model_save_path"]}')  # Print a message indicating that the model and tokenizer have been saved successfully to the specified directory


# ------------ Main Function -------------
# Main function to execute the training process
if __name__ == "__main__":  # Run the following code only if this script is executed as the main program (not imported as a module)
    print('Loading data...')
    train_loader, val_loader, test_loader = load_data(CONFIG)  # Load the training, validation, and testing data 
    print('Data loaded successfully.')

    print('Loading model...')
    model = load_model(CONFIG)  # Load the pre-trained model
    print('Model loaded successfully.')

    print('Starting training...')
    trained_model = train(model, train_loader, val_loader, CONFIG)  # Train the model using the training and validation data
    print('Training completed.')

    print('Saving model...')
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])  # Load the tokenizer to save it along with the model
    save_model(trained_model, tokenizer, CONFIG)  # Save the trained model and tokenizer to the specified directory
    print('Model saved successfully.')

