import os
import argparse
from .model import load_pretrained_model
from .mm_utils import get_model_name_from_path

def main():
    parser = argparse.ArgumentParser(description="Load a pretrained model and lora weight and saveit to the specified path.")
    parser.add_argument("--model_path", required=True, help="Path to the lora weights.")
    parser.add_argument("--save_path", required=True, help="Path to save the model and tokenizer.")

    args = parser.parse_args()
    model_path = args.model_path
    save_path = args.save_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model_path does not exist: {model_path}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)  

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name=model_name)

    model.save_pretrained(save_path, state_dict=model.state_dict())
    tokenizer.save_pretrained(save_path)

    print(f"Merged model has been saved to: {save_path}")

if __name__ == "__main__":
    main()
