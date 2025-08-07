import argparse
import os

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--save-path", "--save_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    new_state_dict = dict()
    for key, tensor in model.state_dict().items():
        new_key = key.replace("vision_encoder", "vision_encoder.vision_encoder")
        print(f"Convert {key} -> {new_key}")
        new_state_dict[new_key] = tensor
    torch.save(new_state_dict, os.path.join(args.save_path, "pytorch_model.bin"))

    config = model.config.to_dict()
    config["vision_encoder"] = "DAMO-NLP-SG/SigLIP-NaViT"
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    tokenizer.save_pretrained(args.save_path)