import mlflow
import mlflow.sklearn
import pandas as pd


MODEL_NAME = "fraud-detection-model"


# ==================================================
# 1. Get Production Model
# ==================================================

model_uri = f"models:/{MODEL_NAME}@production"

model = mlflow.pyfunc.load_model(model_uri)

print("Production model loaded successfully.")


# ==================================================
# 2. Load the processed fraud dataset
# ==================================================

DATA_PATH = "data/processed/fraud_baseline.csv"

df = pd.read_csv(DATA_PATH)


# ==================================================
# 3. Separate features and target
# ==================================================

X = df.drop(columns=["isFraud"])

y = df["isFraud"]


# ==================================================
# 4. Take one real transaction
# ==================================================

sample = X.iloc[[0]]

actual = y.iloc[0]


# ==================================================
# 5. Prediction
# ==================================================

prediction = model.predict(sample)

print("\n================================")
print("FRAUD PREDICTION")
print("================================")

print("Prediction:", int(prediction[0]))
print("Actual    :", int(actual))


# ==================================================
# 6. Load sklearn model to get probability
# ==================================================

sklearn_model = mlflow.sklearn.load_model(model_uri)

probability = sklearn_model.predict_proba(sample)

fraud_probability = probability[0][1]

print(
    "Fraud Probability:",
    round(float(fraud_probability), 4)
)


# ==================================================
# 7. Final result
# ==================================================

if prediction[0] == 1:
    print("Result    : FRAUD")
else:
    print("Result    : LEGITIMATE")