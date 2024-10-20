import json

from datasets import load_dataset
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

# load config

with open("./finetuning_open_source_llms/project_1/config.json", "r") as f:
    config = json.load(f)

api_key = config["API_KEY"]
api_url = config["API_URL"]
dataset_name = "davidhornshaw-DIBT_10k_prompts"  # temp
local_model = config["LOCAL_MODEL"]
workspace = config["WORKSPACE"]

hf_repository = config["HF_REPOSITORY"]
hf_token = config["HF_TOKEN"]

# load and filter dataset according to highest quality responses

hf_dataset = load_dataset("DIBT/10k_prompts_ranked", split="train")

filtered_dataset = hf_dataset.filter(
    lambda r: float(r["avg_rating"]) >= 4 and int(r["num_responses"]) >= 2
)

filtered_dataset_red = filtered_dataset.select(range(0, 64))  # reduce compute

# create distilab pipeline

with Pipeline(
    name="prefs-with-tinyllama",
    description="Pipeline creating preference datasets using TinyLlama.",
) as pipeline:

    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=filtered_dataset_red,
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=TransformersLLM(
            model=local_model,
            device_map="auto",  # selects available GPUs efficiently
            torch_dtype="auto",  # selects suitable dtype for model
            trust_remote_code=True,
            model_kwargs={
                "low_cpu_mem_usage": True,
            },
        ),
    )

    keep_columns = KeepColumns(
        name="keep_columns",
        columns=["instruction", "generation"],
    )

    # pipeline is supposed to push directly; however TextGeneration results
    # in None values; it is unclear how these can be handled in-pipeline;
    # therefore, we reroute through huggingface; note this invalidates
    # need for container dataset as per setup and should have a fix

    # to_argilla = TextGenerationToArgilla(
    #    name="to_argilla",
    #    dataset_name=dataset_name,
    #    dataset_workspace=workspace,
    # )

    load_dataset >> text_generation >> keep_columns  # >> to_argilla

# run distilab pipeline

distiset = pipeline.run(
    use_cache=False,
    parameters={
        "load_dataset": {
            "batch_size": 16,
        },
        "text_generation": {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.95,
                    "top_k": 50,
                }
            }
        },
        # same issue as above
        # "to_argilla": {
        #    "api_url": api_url,
        #    "api_key": api_key,
        #    "dataset_name": dataset_name,
        #    "dataset_workspace": workspace,
        # },
    },
)

# push distiset to huggingface in preparation for rerouting

distiset.push_to_hub(
    hf_repository,
    commit_message="Pipeline rerouting.",
    private=False,
    token=hf_token,
)

print(f"Dataset '{dataset_name}' pushed to Hugginface.")
