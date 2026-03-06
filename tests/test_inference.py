import pytest
import torch
from transformers import DistilBertTokenizer
from src.train import CONFIG, load_model
from src.inference import load_model_for_inference, preprocess_for_inference, predict_article_violated, format_predictions, LABEL_NAMES, ARTICLE_DESCRIPTIONS


# -------- COnstants for testing --------
NUM_LABELS = CONFIG["num_labels"]
MAX_LENGTH = CONFIG["max_length"]

# --------- Sample legal texts for testing --------
SHORT_TEXT = "The applicant was detained without trial for several months."
LONG_TEXT = "The applicant was detained without trial for several months. " \
            "During this time, they were subjected to inhumane treatment and denied access to legal representation. " \
            "The conditions of detention were deplorable, and the applicant's health deteriorated significantly. " \
            "This case raises serious concerns about the violation of fundamental human rights and the rule of law."
LIST_TEXT = [
    "The applicant was detained without trial for several months.",
    "No judge was appointed to review the case.",
    "The domestic courts failed to provide an effective remedy for the applicant's situation."
]

# --------- Fixtures ----------------------
@pytest.fixture(scope="module")
def tokenizer():
    return DistilBertTokenizer.from_pretrained(CONFIG["model_name"]) # use model_name not model_save_path to load the tokenizer as testing should not depend on the saved model

@pytest.fixture(scope="module")
def model():
    return load_model(CONFIG)

@pytest.fixture(scope="module")
def short_text_encoding(short_text, tokenizer):
    return preprocess_for_inference(SHORT_TEXT, tokenizer, CONFIG)

@pytest.fixture(scope="module")
def predictions_short(model, tokenizer):
    return predict_article_violated(SHORT_TEXT, model, tokenizer, CONFIG)

@pytest.fixture(scope="module")
def predictions_long(model, tokenizer):
    return predict_article_violated(LONG_TEXT, model, tokenizer, CONFIG)

@pytest.fixture(scope="module")
def predictions_list(model, tokenizer):
    return predict_article_violated(LIST_TEXT, model, tokenizer, CONFIG)

# --------- Test cases ----------------------

#--------- For preprocessing ---------
def test_preprocess_string_input(short_text_encoding):
    ''' String input should be tokenized and encoded correctly '''
    encoding = short_text_encoding
    assert "input_ids" in encoding and "attention_mask" in encoding, "Encoding should contain input_ids and attention_mask"
    assert encoding["input_ids"].shape == (1, MAX_LENGTH), f"Expected input_ids shape (1, {MAX_LENGTH}), but got {encoding['input_ids'].shape}"
    assert encoding["attention_mask"].shape == (1, MAX_LENGTH), f"Expected attention_mask shape (1, {MAX_LENGTH}), but got {encoding['attention_mask'].shape}"

def test_preprocess_list_input(tokenizer):
    ''' List of strings input should be tokenized and encoded correctly '''
    encoding = preprocess_for_inference(LIST_TEXT, tokenizer, CONFIG)
    assert "input_ids" in encoding and "attention_mask" in encoding, "Encoding should contain input_ids and attention_mask"
    assert encoding["input_ids"].shape == (len(LIST_TEXT), MAX_LENGTH), f"Expected input_ids shape ({len(LIST_TEXT)}, {MAX_LENGTH}), but got {encoding['input_ids'].shape}"
    assert encoding["attention_mask"].shape == (len(LIST_TEXT), MAX_LENGTH), f"Expected attention_mask shape ({len(LIST_TEXT)}, {MAX_LENGTH}), but got {encoding['attention_mask'].shape}"

def test_preprocess_empty_string(tokenizer):
    ''' Empty string input should be handled gracefully '''
    encoding = preprocess_for_inference("", tokenizer, CONFIG)
    assert "input_ids" in encoding and "attention_mask" in encoding, "Encoding should contain input_ids and attention_mask"
    assert encoding["input_ids"].shape == (1, MAX_LENGTH), f"Expected input_ids shape (1, {MAX_LENGTH}), but got {encoding['input_ids'].shape}"
    assert encoding["attention_mask"].shape == (1, MAX_LENGTH), f"Expected attention_mask shape (1, {MAX_LENGTH}), but got {encoding['attention_mask'].shape}"

def test_preprocess_long_text_truncation(tokenizer):
    ''' Long text input should be truncated to max_length '''
    encoding = preprocess_for_inference(LONG_TEXT, tokenizer, CONFIG)
    assert encoding["input_ids"].shape == (1, MAX_LENGTH), f"Expected input_ids shape (1, {MAX_LENGTH}), but got {encoding['input_ids'].shape}"
    assert encoding["attention_mask"].shape == (1, MAX_LENGTH), f"Expected attention_mask shape (1, {MAX_LENGTH}), but got {encoding['attention_mask'].shape}"

#---------- For prediction ---------
def test_predict_returns_dict(predictions_short):
    ''' The predict_article_violated function should return a dictionary of predicted labels and their probabilities '''
    assert isinstance(predictions_short, dict), f"Expected output to be a dictionary, but got {type(predictions_short)}"

def test_predict_result_contains_valid_labels(predictions_short):
    ''' The predicted labels should be valid article labels from the ECtHR dataset '''
    for label in predictions_short.keys():
        assert label in LABEL_NAMES, f"Predicted label {label} is not a valid label. Expected one of {LABEL_NAMES}"

def test_predict_probabilities_in_range(predictions_short):
    ''' The probabilities in the prediction results should be between o and 1'''
    for article, result in predictions_short.items():
        prob = result["probability"]
        assert 0 <= prob <= 1, f"Probability for article {article} is out of range: {prob}. Expected a value between 0 and 1."

def test_predict_only_above_threshold(predictions_short):
    ''' The predict_article_violated function should only return labels whose probabilities are above the threshold '''
    for article, result in predictions_short.items():
        prob = result["probability"]
        assert prob >= CONFIG["threshold"], f"Predicted probability for article {article} is below the threshold: {prob}. Expected a value above {CONFIG['threshold']}"

def test_predict_list_input(predictions_list):
    ''' The predict_article_violated function should handle list input and return predictions accordingly '''
    assert isinstance(predictions_list, dict), f"Expected output to be a dictionary, but got {type(predictions_list)}"
    for label in predictions_list.keys():
        assert label in LABEL_NAMES, f"Predicted label {label} is not a valid label. Expected one of {LABEL_NAMES}"

def test_predict_none_input(model, tokenizer):
    """None input should raise an error."""
    with pytest.raises(Exception):
        predict_article_violated(None, model, tokenizer, CONFIG)

def test_predict_empty_string_input(model, tokenizer):
    """Empty string input should return no predictions."""
    result = predict_article_violated("", model, tokenizer, CONFIG)
    assert isinstance(result, dict), f"Expected output to be a dictionary, but got {type(result)}"

# ----------- For format_predictions ---------
def test_format_predictions_no_results(capsys):
    ''' The format_predictions function should print a message when there are no predictions '''
    format_predictions({})
    captured = capsys.readouterr()
    assert "No articles predicted as violated." in captured.out, f"Expected message about no predictions, but got: {captured.out}"

def test_format_predictions_with_results(capsys, predictions_short):
    ''' The format_predictions function should print the predicted articles and their descriptions in a readable format '''
    format_predictions(predictions_short)
    captured = capsys.readouterr()
    if predictions_short:
        assert "Predicted Article Violations" in captured.out, f"Expected header for predicted articles, but got: {captured.out}"

##########################
# capsys is a builtin pytest fixture, stands for "capture system outputs". 
# It allows you to capture and test the output printed to the console (stdout) and error messages (stderr) during the execution of a test function.
# capsys.readouterr() returns a tuple containing the captured stdout and stderr as strings, which you can then assert against expected values in your tests.
