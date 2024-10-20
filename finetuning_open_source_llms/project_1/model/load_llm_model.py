import json

from huggingface_hub import snapshot_download

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

local_model = config["LOCAL_MODEL"]

# download model

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
snapshot_download(repo_id=model_name, local_dir=local_model)
