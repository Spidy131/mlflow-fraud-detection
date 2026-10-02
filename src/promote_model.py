import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "fraud-detection-model"
EXPERIMENT_NAME = "fraud-detection"


client = MlflowClient()


# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")


# Find best run based on F1 score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found.")


best_run = runs[0]

run_id = best_run.info.run_id
f1_score = best_run.data.metrics["f1_score"]
model_name = best_run.data.params.get("model_name")


print("Best model:", model_name)
print("F1 score:", f1_score)
print("Run ID:", run_id)


# Register model if not already registered
model_uri = f"runs:/{run_id}/model"

try:
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
except Exception:
    versions = []


if not versions:
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    version = registered.version

else:
    # Find version created from the best run
    version = None

    for v in versions:
        if v.run_id == run_id:
            version = v.version
            break

    if version is None:
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        version = registered.version


print("Registered version:", version)


# --------------------------------------------------
# Promote to Production
# --------------------------------------------------

client.set_registered_model_alias(
    MODEL_NAME,
    "production",
    version
)


print("\n================================")
print("MODEL PROMOTED")
print("================================")

print("Model :", MODEL_NAME)
print("Version:", version)
print("Alias : production")