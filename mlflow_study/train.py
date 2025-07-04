# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import mlflow
from mlflow_config import MLflowConfig

#%%
# Configuração do MLflow
load_dotenv()
# Configuração do MLflow Tracking
config = MLflowConfig()


mlflow_url = config.get_mlflow_tracking_uri()

#%%
# Configurar tracking URI
if mlflow_url:
    mlflow.set_tracking_uri(mlflow_url)
else:
    print("Aviso: MLFLOW_URI não configurado. Usando local tracking.")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

mlflow.set_experiment(experiment_name="churn_mlflow_study_tmw")

# %%
# Carregamento dos dados
df = pd.read_csv("data/abt.csv")

# Seleção das features e target
features = df.columns[2:-1]
target = 'flag_churn'

X = df[features]
y = df[target]

# %%
# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=0.2,
                                    random_state=42,
                                    stratify=y)

print("A taxa de resposta no treino é: ", y_train.mean())
print("A taxa de resposta no teste é: ", y_test.mean())

# %%
# Treinamento do modelo com mlflow

with mlflow.start_run():

    mlflow.sklearn.autolog()
    
    # model = DecisionTreeClassifier(
    #     min_samples_leaf=100,
    #     min_samples_split=10,
    #     random_state=42)

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=120,
        min_samples_split=20,
        random_state=42)

    model.fit(X_train, y_train)

    # Predição
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)


    mlflow.log_metric("accuracy_test", accuracy_test)


# %%

print("Acurácia no treino: ", accuracy_train)
print("Acurácia no teste: ", accuracy_test)

# %%
