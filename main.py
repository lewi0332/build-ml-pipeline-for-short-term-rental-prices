import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    root_path = hydra.utils.get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                # os.path.join(root_path, "components/get_data"), #NOTE github URL not working.
                "main",
                version="main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "cleaned.csv",
                    "output_artifact_type": "cleaned_data",
                    "output_artifact_description": "Cleaned file"
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                parameters={
                    "input_artifact": "cleaned.csv:latest",
                    "output_artifact": "data_check.json",
                    "output_artifact_type": "data_check",
                    "output_artifact_description": "Data check"
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_split",
                "main",
                parameters={
                    "input_artifact": "cleaned.csv:latest",
                    "output_artifact": "train.csv",
                    "output_artifact_type": "train_data",
                    "output_artifact_description": "Train data",
                    "output_artifact_2": "test.csv",
                    "output_artifact_type_2": "test_data",
                    "output_artifact_description_2": "Test data",
                    "test_size": config["modeling"]["test_size"],
                    "random_state": config["modeling"]["random_state"]
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_random_forest",
                "main",
                parameters={
                    "input_artifact": "train.csv:latest",
                    "output_artifact": "random_forest.pkl",
                    "output_artifact_type": "random_forest_model",
                    "output_artifact_description": "Random forest model",
                    "rf_config": rf_config
                },
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                parameters={
                    "input_artifact": "random_forest.pkl:prod",
                    "input_artifact_2": "test.csv:latest",
                    "output_artifact": "test_regression_model.json",
                    "output_artifact_type": "test_regression_model",
                    "output_artifact_description": "Test regression model"
                },
            )

if __name__ == "__main__":
    go()
