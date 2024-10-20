import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

local_model = config["LOCAL_MODEL"]

# load tokenizer and model to verify download

tokenizer = AutoTokenizer.from_pretrained(local_model)
model = AutoModelForCausalLM.from_pretrained(local_model)

# print size of model

filepath = Path(local_model).glob("**/*")
total_size = sum(f.stat().st_size for f in filepath if f.is_file())

print(f"Total size of downloaded model: {total_size / 1e9:.2f} GB.")
