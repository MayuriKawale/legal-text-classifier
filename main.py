import argparse
from src.train import CONFIG

#------- Argument parsing for command-line interface -------
parser = argparse.ArgumentParser(description="Legal Text Classifier for ECtHR dataset") # create an instance of ArgumentParser to handle command-line arguments
parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "inference"], help="Mode to run the script: train, evaluate, or inference") # add a required argument --mode
parser.add_argument("--text", type=str, default=None, help="Input legal text for inference mode") # add an optional argument --text
args = parser.parse_args() # parse the command-line arguments

#------- Actions based on the mode argument -------
if args.mode == "train":
    from src.train import load_data, load_model, train, save_model
    from transformers import DistilBertTokenizer
    train_loader, val_loader, test_loader = load_data(CONFIG) 
    model = load_model(CONFIG)
    model = train(model, train_loader, val_loader, CONFIG)
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])
    save_model(model, tokenizer, CONFIG)

elif args.mode == "evaluate":
    from src.evaluate import load_saved_model, evaluate
    from src.train import ECtHRDataset
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    model, tokenizer = load_saved_model(CONFIG)
    raw_dataset = load_dataset("lex_glue", "ecthr_a")
    test_dataset = ECtHRDataset(raw_dataset["test"], tokenizer, CONFIG)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    evaluate(model, test_loader, CONFIG)

elif args.mode == "inference":
    from src.inference import load_model_for_inference, predict_article_violated, format_predictions
    if args.text is None:
        raise ValueError("In inference mode, please provide input text using the --text argument.")
    else:
        model, tokenizer = load_model_for_inference(CONFIG)
        predictions = predict_article_violated(args.text, model, tokenizer, CONFIG)
        format_predictions(predictions)
        