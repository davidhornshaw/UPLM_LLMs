import json

import argilla as rg

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

api_key = config["API_KEY"]
api_url = config["API_URL"]
workspace = config["WORKSPACE"]

# connect to public argilla instance

client = rg.Argilla(
    api_url=api_url,
    api_key=api_key,
)

# create project workspace

rg.Workspace(name=workspace).create()

print(f"New workspace '{workspace}' created.")
