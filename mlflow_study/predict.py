# %%
import mlflow

from dotenv import load_dotenv
from mlflow_config import MLflowConfig
import os

# %%

# Configuração do MLflow
load_dotenv()
# Configuração do MLflow Tracking
config = MLflowConfig()


mlflow_url = config.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_url)

# %%
client = mlflow.client.MlflowClient()
model_versions = client.search_model_versions("name='churn-tmw-study'")
latest_version = max(model_versions, key=lambda x: int(x.version))

# %%
model = mlflow.sklearn.load_model(f"models:/churn-tmw-study/{latest_version.version}")

# %%
model.feature_names_in_
# %%


import pandas as pd
df = pd.read_csv("data/abt.csv")
df.head()
# %%
X = df.head()[model.feature_names_in_]
# %%

proba = model.predict_proba(X)
proba
# %%
