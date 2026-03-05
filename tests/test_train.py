import pytest
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from datasets import load_dataset
from src.train import ECtHRDataset, load_model, CONFIG


# To run these tests, use the command: pytest tests/test_train.py -v (v is for verbose output)

## Comment on decorators used in the code:
# @pytest.fixture(scope="module") creates a decorator that defines a fixture with module-level scope. 
# This means that the fixture will be set up once for the entire test module, and all tests within that module 
# will share the same instance of the fixture. In this case, we have three fixtures: tokenizer, raw_dataset, and train_dataset. 
# Each of these fixtures is initialized once per test module, allowing us to reuse the same tokenizer and dataset across multiple 
# tests without reloading them each time.

@pytest.fixture(scope="module")
def tokenizer():
    return DistilBertTokenizer.from_pretrained(CONFIG["model_name"])


@pytest.fixture(scope="module")
def raw_dataset():
    return load_dataset("lex_glue", "ecthr_a")


@pytest.fixture(scope="module")
def train_dataset(raw_dataset, tokenizer):
    return ECtHRDataset(raw_dataset['train'], tokenizer, CONFIG)


def test_dataset_length(train_dataset):
    """Dataset should have 9000 training examples."""
    assert len(train_dataset) == 9000


def test_example_shapes(train_dataset):
    """Single example should have correct tensor shapes."""
    example = train_dataset[0]
    assert example["input_ids"].shape == torch.Size([512])
    assert example["attention_mask"].shape == torch.Size([512])
    assert example["labels"].shape == torch.Size([10])


def test_labels_are_multihot(train_dataset):
    """Labels should only contain 0s and 1s."""
    example = train_dataset[0]
    assert set(example["labels"].tolist()).issubset({0.0, 1.0})


def test_input_ids_truncated(train_dataset):
    """Input ids should never exceed max_length."""
    example = train_dataset[0]
    assert len(example["input_ids"]) <= CONFIG["max_length"]


def test_dataloader_batch_shape(train_dataset):
    """Batch should have correct shape with batch_size=2."""
    loader = DataLoader(train_dataset, batch_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == torch.Size([2, 512])
    assert batch["attention_mask"].shape == torch.Size([2, 512])
    assert batch["labels"].shape == torch.Size([2, 10])


def test_forward_pass(train_dataset):
    """Forward pass should return loss and logits of correct shape."""
    
    loader = DataLoader(train_dataset, batch_size=2)
    batch = next(iter(loader))

    model = load_model(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )

    assert outputs.loss.item() > 0
    assert outputs.logits.shape == torch.Size([2, 10])