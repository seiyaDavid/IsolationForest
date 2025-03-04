import pandas as pd
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

"""
Data Loading and Preprocessing Module

This module handles all data-related operations including:
    - Loading stock data from CSV files
    - Data validation and cleaning
    - Feature preparation for model training
    - Data formatting for inference
"""


class DataLoader:
    """
    Handles data loading and preprocessing for stock anomaly detection.

    The class provides methods for:
        - Loading stock data
        - Preparing features for model training
        - Validating data format and content
        - Converting data for model inference
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load the stock data from csv file"""
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_stock_data(
        self, df: pd.DataFrame, stock: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for a specific stock"""
        try:
            logger.info(f"Preparing data for stock: {stock}")
            if stock not in df.columns:
                raise ValueError(f"Stock {stock} not found in dataframe")

            X = df[[stock]].copy()
            dates = df["Date"]
            logger.info(f"Successfully prepared data for {stock}")
            return X, dates
        except Exception as e:
            logger.error(f"Error preparing data for {stock}: {str(e)}")
            raise
