import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def download_and_load_model(model_name="t5-small", device="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Downloading and loading the {model_name} model on {device}...")

    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()  # set to evaluation mode

    return model, tokenizer, device
