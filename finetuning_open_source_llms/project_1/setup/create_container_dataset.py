import json

import argilla as rg

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

api_key = config["API_KEY"]
api_url = config["API_URL"]
dataset_name = config["DATASET_NAME"]
workspace = config["WORKSPACE"]

# connect to public argilla instance

client = rg.Argilla(
    api_url=api_url,
    api_key=api_key,
)

# push distilabel-compliant but empty dataset to workspace

dataset = rg.Dataset(name=dataset_name, workspace=workspace, client=client)

dataset.settings.fields = [
    rg.TextField(name="instruction"),
    rg.TextField(name="generation"),
]

dataset.settings.questions = [
    rg.LabelQuestion(
        name="quality",
        title="How do you evaluate the response quality?",
        labels=["👎", "👍"],
    )
]

dataset.create()

print(f"New dataset '{dataset_name}' created in workspace '{workspace}'.")
