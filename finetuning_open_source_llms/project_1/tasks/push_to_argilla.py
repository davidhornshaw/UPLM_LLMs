import json

import argilla as rg
from datasets import load_dataset

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

api_key = config["API_KEY"]
api_url = config["API_URL"]
dataset_name = "davidhornshaw-DIBT_10k_prompts"  # temp
workspace = config["WORKSPACE"]

hf_repository = config["HF_REPOSITORY"]
hf_token = config["HF_TOKEN"]

# connect to public argilla instance

client = rg.Argilla(
    api_url=api_url,
    api_key=api_key,
)

# pull dataset from huggingface

hf_dataset = load_dataset("davidhornshaw/DIBT_10k_prompts", split="train")

# push filtered dataset to huggingface

filtered_dataset = hf_dataset.filter(
    lambda r: r["generation"] is not None and len(r["generation"]) > 0
)

filtered_dataset.push_to_hub(
    hf_repository,
    commit_message="Pipeline rerouting.",
    private=False,
    token=hf_token,
)

# pull filtered dataset to argilla from huggingface

settings = rg.Settings(
    fields=[
        rg.TextField(name="instruction"),
        rg.TextField(name="generation"),
    ],
    questions=[
        rg.LabelQuestion(
            name="quality",
            labels=["👎", "👍"],
            title="How do you evaluate the response quality?",
        )
    ],
)

# apparently, this ignores/cannot handle workspace;

rg.Dataset.from_hub(hf_repository, settings=settings, workspace_name=workspace)

print(f"Dataset '{dataset_name}' updated in workspace '{workspace}'.")
