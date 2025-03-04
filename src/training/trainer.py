import mlflow
import pandas as pd
import yaml
from src.models.isolation_forest import StockAnomalyDetector
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("trainer")

"""
Stock Anomaly Detection Model Trainer

This module handles the training of anomaly detection models for stock data.
It implements:
    - Isolation Forest algorithm
    - Hyperparameter optimization using Optuna
    - Model persistence with MLflow
    - Training workflow management
"""


class ModelTrainer:
    """
    Handles the training of anomaly detection models for stock data.

    Attributes:
        config (dict): General configuration parameters
        hyperparameters (dict): Model hyperparameter ranges
        mlflow_manager (MLFlowManager): MLflow interface for model management

    The trainer:
        - Optimizes hyperparameters for each stock
        - Trains Isolation Forest models
        - Saves models and metadata to MLflow
        - Handles the complete training workflow
    """

    def __init__(self, config_path: str, hp_config_path: str):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            self.hp_config_path = hp_config_path
            logger.info("ModelTrainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ModelTrainer: {str(e)}")
            raise

    def train_stock_model(self, stock: str, data: pd.DataFrame) -> pd.DataFrame:
        """Train model for a specific stock"""
        try:
            logger.info(f"Starting training process for {stock}")
            data_loader = DataLoader(None)
            X, dates = data_loader.prepare_stock_data(data, stock)

            detector = StockAnomalyDetector(stock, self.hp_config_path)

            with mlflow.start_run(run_name=f"{stock}_anomaly_detection") as run:
                mlflow.set_tag("stock", stock)  # Add stock as a tag
                model, best_params = detector.train(X)

                logger.info(f"Logging parameters for {stock}")
                mlflow.log_params(best_params)

                logger.info(f"Saving model for {stock}")
                mlflow.sklearn.log_model(model, "model")

                predictions = model.predict(X)
                anomaly_scores = model.score_samples(X)

                anomalies = pd.DataFrame(
                    {
                        "Date": dates[predictions == -1],
                        "Stock": stock,
                        "Anomaly_Score": anomaly_scores[predictions == -1],
                        "Value": X.iloc[predictions == -1, 0].values,
                    }
                )

                logger.info(f"Found {len(anomalies)} anomalies for {stock}")
                return anomalies, predictions, anomaly_scores
        except Exception as e:
            logger.error(f"Error in train_stock_model for {stock}: {str(e)}")
            raise
