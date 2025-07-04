# MLflow Study - Projeto de Estudo com MLflow

Este projeto demonstra a implementação de um sistema completo de Machine Learning utilizando MLflow para tracking de experimentos, versionamento de modelos e deploy. O projeto foca na predição de churn de clientes usando algoritmos de classificação.

## Descrição

Este é um projeto de estudo que implementa um pipeline completo de Machine Learning com as seguintes funcionalidades:

- **Tracking de Experimentos**: Registro automático de parâmetros, métricas e artefatos
- **Versionamento de Modelos**: Controle de versão dos modelos treinados
- **Integração com MinIO/S3**: Armazenamento de artefatos em storage compatível com S3
- **Predição em Tempo Real**: Sistema de inferência usando modelos versionados
- **Visualização de Resultados**: Métricas e gráficos de performance

## Arquitetura

```
mlflow_study/
├── mlflow_study/
│   ├── __init__.py
│   ├── train.py              # Script de treinamento
│   ├── predict.py            # Script de predição
│   ├── mlflow_config.py      # Configuração do MLflow
│   ├── data/
│   │   └── abt.csv           # Dataset de churn
│   └── mlruns/               # Diretório local de experimentos
├── pyproject.toml            # Dependências do projeto
├── poetry.lock              # Lock file do Poetry
├── example.env              # Exemplo de variáveis de ambiente
└── README.md                
```

## Instalação

### Pré-requisitos

- Python 3.13+
- Poetry (gerenciador de dependências)

### Passos de Instalação

1. **Clone o repositório**:
```bash
git clone <url-do-repositorio>
cd mlflow_study
```

2. **Instale as dependências**:
```bash
poetry install
```

3. **Configure as variáveis de ambiente**:
```bash
cp example.env .env
# Edite o arquivo .env com suas configurações
```

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` baseado no `example.env`:

```env
# MLflow Tracking Server
MLFLOW_TRACKING_URI=hostname:3000

# Configuração do MinIO (S3-compatible)
AWS_ACCESS_KEY_ID=username
AWS_SECRET_ACCESS_KEY=password
AWS_DEFAULT_REGION=us-east-1
MLFLOW_S3_ENDPOINT_URL=localhost:9002
MLFLOW_S3_IGNORE_TLS=true
MLFLOW_S3_BUCKET=mlflow
MLFLOW_S3_USE_HTTPS=false
```

### Configuração do MLflow

O projeto utiliza uma classe `MLflowConfig` para gerenciar as configurações:

- **Tracking URI**: Endereço do servidor MLflow
- **MinIO/S3**: Configuração para armazenamento de artefatos
- **Credenciais AWS**: Para acesso ao storage S3-compatible

## Dataset

O projeto utiliza um dataset de churn (`data/abt.csv`) oriundo do TeoMeWhy com as seguintes características:

- **Target**: `flag_churn` (variável binária)
- **Features**: Múltiplas variáveis numéricas e categóricas
- **Divisão**: 80% treino / 20% teste (estratificada)

## Uso

### 1. Treinamento do Modelo

Execute o script de treinamento:

```bash
cd mlflow_study
poetry run python train.py
```

**Funcionalidades do treinamento**:
- Carregamento automático dos dados
- Divisão estratificada treino/teste
- Treinamento com Random Forest
- Log automático de parâmetros e métricas
- Salvamento do modelo no MLflow Model Registry

**Parâmetros do modelo atual**:
- `n_estimators`: 500
- `min_samples_leaf`: 120
- `min_samples_split`: 20
- `random_state`: 42

### 2. Predição

Execute o script de predição:

```bash
poetry run python predict.py
```

**Funcionalidades da predição**:
- Carregamento automático do último modelo versionado
- Predição em novos dados
- Retorno de probabilidades de churn

### 3. Visualização dos Resultados

Acesse a interface web do MLflow:

```bash
mlflow ui
```

**Métricas disponíveis**:
- Acurácia no treino e teste
- Matriz de confusão
- Curva ROC
- Curva Precision-Recall

## Experimentos

### Estrutura dos Experimentos

- **Nome do Experimento**: `churn_mlflow_study_tmw`
- **Modelo Registry**: `churn-tmw-study`
- **Artefatos Salvos**:
  - Modelo serializado (`.pkl`)
  - Gráficos de performance
  - Configurações do ambiente

### Tracking Automático

O MLflow registra automaticamente:
- Parâmetros do modelo
- Métricas de performance
- Artefatos (gráficos, modelo)
- Configuração do ambiente

## Dependências

### Principais Bibliotecas

- **MLflow** (>=3.1.1): Tracking de experimentos e model registry
- **Scikit-learn** (>=1.7.0): Algoritmos de Machine Learning
- **Pandas** (>=2.3.0): Manipulação de dados
- **Boto3** (>=1.39.2): Integração com AWS/MinIO
- **Psycopg2** (>=2.9.10): Conectividade com PostgreSQL

### Ferramentas de Desenvolvimento

- **Poetry**: Gerenciamento de dependências
- **Ruff**: Linter e formatação de código
- **IPykernel**: Suporte a Jupyter notebooks

## 🔍 Estrutura do Código

### `train.py`
- Configuração do MLflow
- Carregamento e preparação dos dados
- Treinamento do modelo
- Log de métricas e artefatos

### `predict.py`
- Carregamento do modelo versionado
- Predição em novos dados
- Retorno de probabilidades

### `mlflow_config.py`
- Classe para configuração do MLflow
- Gerenciamento de variáveis de ambiente
- Integração com MinIO/S3

## 📄 Licença

Este projeto é para fins educacionais e de estudo.

---

**Nota**: Este projeto foi desenvolvido como um estudo prático de MLflow e boas práticas em Machine Learning, baseando-se no vídeo em curso do TeoMeWhy no youtube.