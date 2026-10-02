import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-detection-model"

# Metric used for model selection
SELECTION_METRIC = "pr_auc"


client = MlflowClient()


# ==================================================
# 1. Get experiment
# ==================================================

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' not found."
    )


# ==================================================
# 2. Find best run
# ==================================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[
        f"metrics.{SELECTION_METRIC} DESC"
    ],
    max_results=1
)


if not runs:
    raise ValueError(
        "No MLflow runs found."
    )


best_run = runs[0]


# ==================================================
# 3. Extract information
# ==================================================

run_id = best_run.info.run_id

pr_auc = best_run.data.metrics[
    SELECTION_METRIC
]

f1_score = best_run.data.metrics[
    "f1_score"
]

model_name = best_run.data.params.get(
    "model_name"
)


print("\n================================")
print("BEST MODEL")
print("================================")

print("Model   :", model_name)
print("PR-AUC  :", pr_auc)
print("F1      :", f1_score)
print("Run ID  :", run_id)


# ==================================================
# 4. Model URI
# ==================================================

model_uri = f"runs:/{run_id}/model"


# ==================================================
# 5. Register model
# ==================================================

print("\nRegistering model...")

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

version = registered_model.version


print("Registered Version:", version)


# ==================================================
# 6. Set production alias
# ==================================================

client.set_registered_model_alias(
    MODEL_NAME,
    "production",
    version
)


print("\n================================")
print("PRODUCTION MODEL")
print("================================")

print("Model   :", MODEL_NAME)
print("Version :", version)
print("Alias   : production")
print("PR-AUC  :", pr_auc)
print("F1      :", f1_score)