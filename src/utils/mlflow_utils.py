import mlflow
import yaml
from src.utils.logger import setup_logger
import os
from typing import Optional, Any

logger = setup_logger("mlflow_utils")

"""
MLflow Model Management Utilities

This module handles all MLflow-related operations including:
    - Model versioning
    - Model storage and retrieval
    - Experiment tracking
    - Model metadata management
    - Run history tracking

The module ensures consistent model management across training and inference.
"""


class MLFlowManager:
    """
    Manages MLflow operations for model versioning and tracking.

    This class provides a centralized interface for all MLflow operations,
    ensuring consistent model management across the application.

    Attributes:
        config (dict): Configuration parameters for MLflow including:
            - tracking_uri: Location for storing MLflow data
            - experiment_name: Name of the MLflow experiment

    Methods:
        get_model: Retrieve latest model for a specific stock
        log_model: Save a trained model with metadata
        get_run_history: Retrieve training history for a stock
    """

    def __init__(self, config_path: str):
        """
        Initialize MLflow manager with configuration.

        Args:
            config_path (str): Path to YAML configuration file containing:
                mlflow:
                    tracking_uri: Path to MLflow tracking directory
                    experiment_name: Name for the experiment

        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required config keys are missing
        """
        try:
            logger.info(f"Initializing MLFlowManager with config: {config_path}")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            tracking_uri = self.config["mlflow"]["tracking_uri"]
            experiment_name = self.config["mlflow"]["experiment_name"]

            logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)

            logger.info(f"Setting MLflow experiment: {experiment_name}")
            mlflow.set_experiment(experiment_name)

            logger.info("MLFlowManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MLFlowManager: {str(e)}")
            raise

    def get_model(self, stock: str) -> Optional[Any]:
        """
        Retrieve the latest trained model for a specific stock.

        Args:
            stock (str): Stock symbol/name to retrieve model for

        Returns:
            Optional[Any]: Latest trained model if exists, None otherwise

        Note:
            Returns the most recently trained model based on run timestamp
        """
        try:
            logger.info(f"Checking for existing model for {stock}")
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(
                self.config["mlflow"]["experiment_name"]
            )

            if experiment is None:
                logger.info("No experiment found")
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.stock = '{stock}'",
            )

            if runs:
                latest_run = sorted(
                    runs, key=lambda x: x.info.start_time, reverse=True
                )[0]
                model = mlflow.sklearn.load_model(
                    f"runs:/{latest_run.info.run_id}/model"
                )
                logger.info(f"Successfully loaded model for {stock}")
                return model
            return None
        except Exception as e:
            logger.warning(f"Model not found for {stock}: {str(e)}")
            return None
