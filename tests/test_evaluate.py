# Imports
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from src.train import ECtHRDataset, CONFIG
from src.evaluate import get_predictions, evaluate, load_saved_model, LABEL_NAMES

# ------- Fixtures to fix the tokenizer, raw dataset and test data loader -------
@pytest.fixture(scope="module")
def tokenizer():
    return DistilBertTokenizer.from_pretrained(CONFIG["model_name"])

@pytest.fixture(scope="module")
def raw_dataset():
    return load_dataset("lex_glue", "ecthr_a")

@pytest.fixture(scope="module")
def test_loader(tokenizer, raw_dataset):
    test_dataset = ECtHRDataset(raw_dataset["test"], tokenizer, CONFIG)
    return DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# ------- Tests -------

def test_number_of_labels():
    '''Test that the number of labels is correct. The ECtHR dataset has 10 labels.'''
    assert len(LABEL_NAMES) == 10, f"Expected 10 labels, but got {len(LABEL_NAMES)}"

def test_label_names():
    '''Test that the label names are correct. The ECtHR dataset has specific labels.'''
    expected_labels = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']
    assert LABEL_NAMES == expected_labels, f"Expected label names {expected_labels}, but got {LABEL_NAMES}"

def test_saved_model_path_exists():
    '''Test that the saved model path exists.'''
    import os
    assert os.path.exists(CONFIG["model_save_path"]), f"Model save path {CONFIG['model_save_path']} does not exist"

def test_predictions_shape(test_loader):
    '''Test that the shape of the predictions matches the shape of the labels.'''
    # Use untrained model for testing the shape of predictions
    model = DistilBertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=len(LABEL_NAMES))
    model.eval()  # set the model to evaluation mode
    all_preds, all_labels = get_predictions(model, test_loader, CONFIG)

    assert all_preds.shape == (1000, 10) # the test set has 1000 samples and 10 labels
    assert all_labels.shape == (1000, 10) # the test set has 1000 samples and 10 labels

def test_predictions_labels_are_binary(test_loader):
    '''Test that the predictions are binary (0 or 1).'''
    # Use untrained model for testing the binary nature of predictions
    model = DistilBertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=len(LABEL_NAMES))
    model.eval()  # set the model to evaluation mode
    all_preds, all_labels = get_predictions(model, test_loader, CONFIG)

    assert set(np.unique(all_preds)).issubset({0, 1})
    assert set(np.unique(all_labels)).issubset({0, 1})

def test_evaluate_f1_range(test_loader):
    """All F1 scores should be between 0 and 1."""
    from src.train import load_model
    model = load_model(CONFIG)
    metrics = evaluate(model, test_loader, CONFIG)

    assert 0 <= metrics["micro_f1"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1
    for score in metrics["per_label_f1"].values():
        assert 0 <= score <= 1


def test_per_label_f1_has_all_labels(test_loader):
    """Per label F1 should have scores for all 10 articles."""
    from src.train import load_model
    model = load_model(CONFIG)
    metrics = evaluate(model, test_loader, CONFIG)
    assert set(metrics["per_label_f1"].keys()) == set(LABEL_NAMES)





