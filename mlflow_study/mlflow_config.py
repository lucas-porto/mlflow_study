import os
from typing import Optional
import mlflow

class MLflowConfig:
    """
    Classe para configurar MLflow Tracking e MinIO/S3 compatível.
    """
    
    def __init__(
        self,
        mlflow_db_url: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        s3_endpoint_url: Optional[str] = None,
        s3_bucket: str = "mlflow",
        s3_ignore_tls: bool = True,
        verbose: bool = True
    ):
        self.mlflow_db_url = mlflow_db_url or os.getenv("MLFLOW_TRACKING_URI")
        self.aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = aws_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.s3_endpoint_url = s3_endpoint_url or os.getenv("MLFLOW_S3_ENDPOINT_URL")
        self.s3_bucket = s3_bucket or os.getenv("MLFLOW_S3_BUCKET", "mlflow")
        self.s3_ignore_tls = s3_ignore_tls if s3_ignore_tls is not None else (
            os.getenv("MLFLOW_S3_IGNORE_TLS", "true").lower() == "true"
        )
        self.verbose = verbose
        
        self._configure()
    
    def _configure(self):
        """Configura as variáveis de ambiente necessárias."""
        # Configurar credenciais do MinIO/S3
        if self.aws_access_key and self.aws_secret_key:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_key
            os.environ["AWS_DEFAULT_REGION"] = self.aws_region

        if self.s3_endpoint_url:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint_url
            os.environ["MLFLOW_S3_IGNORE_TLS"] = str(self.s3_ignore_tls)
            os.environ["MLFLOW_S3_BUCKET"] = self.s3_bucket
        
        if self.mlflow_db_url:
            os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_db_url
            mlflow.set_tracking_uri(self.mlflow_db_url)
        
        if self.verbose:
            self._print_config()
    
    def _print_config(self):
        """Exibe as configurações atuais (se verbose=True)."""
        print("MLflow and MinIO/S3 Configuration:")
        print(f"MLflow Tracking URI: {self.mlflow_db_url}")
        print(f"S3 Endpoint: {self.s3_endpoint_url}")
        print(f"S3 Bucket: {self.s3_bucket}")
        print(f"AWS Region: {self.aws_region}")
        print(f"Ignore TLS: {self.s3_ignore_tls}")
    
    def get_mlflow_tracking_uri(self) -> Optional[str]:
        """Retorna a URI de tracking do MLflow."""
        return self.mlflow_db_url