"""
Stock Anomaly Detection Streamlit Application

This module provides a web interface for the stock anomaly detection system.
It allows users to:
    - Upload stock price data
    - Train/retrain anomaly detection models
    - Visualize anomalies in interactive plots
    - Download detected anomalies as CSV

The application uses:
    - Streamlit for the web interface
    - MLflow for model management
    - Plotly for interactive visualizations
"""

import streamlit as st
import pandas as pd
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.training.trainer import ModelTrainer
from src.utils.mlflow_utils import MLFlowManager
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger
import os
from typing import List, Dict

logger = setup_logger("streamlit_app")


def create_stock_plot(data, stock, predictions, anomaly_scores):
    """Create plotly figure for a stock"""
    fig = go.Figure()

    # Add original values
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[stock],
            mode="lines",
            name="Original Values",
            line=dict(color="blue"),
        )
    )

    # Add anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    fig.add_trace(
        go.Scatter(
            x=data["Date"].iloc[anomaly_indices],
            y=data[stock].iloc[anomaly_indices],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        )
    )

    fig.update_layout(
        title=f"{stock} Values and Anomalies",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
    )

    return fig


def main():
    """
    Main application function that handles:
        - File upload interface
        - Model training controls
        - Progress tracking
        - Results visualization
        - Anomaly data export

    The function provides two modes:
        1. Force Retrain: Retrain all models regardless of existing ones
        2. Selective Retrain: Only train models for stocks without existing models
    """
    try:
        st.title("Stock Anomaly Detection System")
        logger.info("Starting Streamlit application")

        uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                stocks = [col for col in data.columns if col != "Date"]

                # Create mlruns directory if it doesn't exist
                os.makedirs("mlruns", exist_ok=True)

                try:
                    logger.info("Initializing MLflow manager")
                    mlflow_manager = MLFlowManager("config/config.yaml")
                    logger.info("MLflow manager initialized successfully")

                    logger.info("Initializing model trainer")
                    trainer = ModelTrainer(
                        "config/config.yaml", "config/hyperparameters.yaml"
                    )
                    logger.info("Model trainer initialized successfully")

                    # Initialize results containers
                    all_results = {}
                    all_anomalies = []

                    # Create columns for status and progress
                    status_container = st.empty()
                    progress_bar = st.progress(0)

                    # Create two columns for buttons
                    col1, col2 = st.columns(2)

                    # Force retrain button in first column
                    if col1.button("Force Retrain All Models"):
                        st.warning(
                            "Force retraining all models regardless of existing models..."
                        )
                        all_anomalies = []
                        all_results = {}

                        for idx, stock in enumerate(stocks):
                            status_container.text(
                                f"Force retraining model for {stock}..."
                            )
                            progress_bar.progress((idx + 1) / len(stocks))

                            anomalies, predictions, scores = trainer.train_stock_model(
                                stock, data
                            )
                            if not anomalies.empty:
                                # Add is_anomaly column
                                anomalies["is_anomaly"] = (
                                    True  # Since these are all anomalies
                                )
                                all_anomalies.append(anomalies)
                            all_results[stock] = (predictions, scores)

                        if all_anomalies:
                            final_anomalies = pd.concat(all_anomalies)
                            final_anomalies.to_csv("anomalies.csv", index=False)
                            st.success(
                                "All models force retrained and anomalies saved!"
                            )

                            # Create download button for anomalies file
                            with open("anomalies.csv", "rb") as f:
                                st.download_button(
                                    label="Download Anomalies CSV",
                                    data=f,
                                    file_name="anomalies.csv",
                                    mime="text/csv",
                                    help="Click to download the anomalies data",
                                )

                            # Create and show plots
                            create_and_show_plots(data, stocks, all_results)

                            # Show anomalies dataframe
                            st.dataframe(final_anomalies)

                    # Regular retrain button in second column
                    elif col2.button("Retrain Missing Models Only"):
                        all_anomalies = []
                        all_results = {}
                        stocks_to_train = []

                        # Check which stocks need training
                        for stock in stocks:
                            model = mlflow_manager.get_model(stock)
                            if model is None:
                                stocks_to_train.append(stock)

                        if stocks_to_train:
                            st.warning(
                                f"Training new models for: {', '.join(stocks_to_train)}"
                            )

                            for idx, stock in enumerate(stocks_to_train):
                                status_container.text(f"Training model for {stock}...")
                                progress_bar.progress((idx + 1) / len(stocks_to_train))

                                anomalies, predictions, scores = (
                                    trainer.train_stock_model(stock, data)
                                )
                                if not anomalies.empty:
                                    # Add is_anomaly column
                                    anomalies["is_anomaly"] = (
                                        True  # Since these are all anomalies
                                    )
                                    all_anomalies.append(anomalies)
                                all_results[stock] = (predictions, scores)

                        # Process stocks with existing models
                        for stock in [s for s in stocks if s not in stocks_to_train]:
                            model = mlflow_manager.get_model(stock)
                            X, dates = DataLoader(None).prepare_stock_data(data, stock)
                            predictions = model.predict(X)
                            scores = model.score_samples(X)
                            all_results[stock] = (predictions, scores)

                            anomalies = pd.DataFrame(
                                {
                                    "Date": dates[predictions == -1],
                                    "Stock": stock,
                                    "Anomaly_Score": scores[predictions == -1],
                                    "Value": X.iloc[predictions == -1, 0].values,
                                    "is_anomaly": True,  # Add is_anomaly column
                                }
                            )

                            if not anomalies.empty:
                                all_anomalies.append(anomalies)

                    # Create plots if we have results
                    if all_results:
                        create_and_show_plots(data, stocks, all_results)

                    # Show anomalies if we have any and create download button
                    if all_anomalies:
                        final_anomalies = pd.concat(all_anomalies)
                        final_anomalies.to_csv("anomalies.csv", index=False)
                        st.success("Anomaly detection completed!")

                        # Create download button for anomalies file
                        with open("anomalies.csv", "rb") as f:
                            st.download_button(
                                label="Download Anomalies CSV",
                                data=f,
                                file_name="anomalies.csv",
                                mime="text/csv",
                                help="Click to download the anomalies data",
                            )

                        st.dataframe(final_anomalies)

                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}")
                    st.error(f"Error processing uploaded file: {str(e)}")
            except Exception as e:
                logger.error(f"Error initializing MLflow or trainer: {str(e)}")
                st.error(f"Error initializing MLflow or trainer: {str(e)}")
                return

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


def create_and_show_plots(
    data: pd.DataFrame, stocks: List[str], all_results: Dict
) -> None:
    """
    Create and display interactive plots for all stocks showing values and anomalies.

    Args:
        data (pd.DataFrame): DataFrame containing stock data
        stocks (List[str]): List of stock names to plot
        all_results (Dict): Dictionary containing predictions and scores for each stock
            Format: {stock_name: (predictions, scores)}

    The function creates a grid of subplots, one for each stock, showing:
        - Original stock values as lines
        - Detected anomalies as red points
    """
    if all_results:
        n_stocks = len(stocks)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols

        # Create subplots with more width and height
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=stocks,
            horizontal_spacing=0.1,  # Reduce space between columns
            vertical_spacing=0.2,  # Add more space between rows
        )

        for idx, stock in enumerate(stocks):
            predictions, scores = all_results[stock]
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Add original values
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[stock],
                    mode="lines",
                    name=f"{stock} Values",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            fig.add_trace(
                go.Scatter(
                    x=data["Date"].iloc[anomaly_indices],
                    y=data[stock].iloc[anomaly_indices],
                    mode="markers",
                    name=f"{stock} Anomalies",
                    marker=dict(color="red", size=10),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Update layout with larger size and better formatting
        fig.update_layout(
            height=400 * n_rows,  # Increased height per row
            width=1400,  # Increased overall width
            title_text="Stock Values and Anomalies",
            showlegend=False,
            title_x=0.5,  # Center the title
            margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins
        )

        # Update all x-axes to show better date formatting
        for i in range(1, n_rows * n_cols + 1):
            fig.update_xaxes(
                tickangle=45, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1
            )

        # Add a unique key to the plotly chart
        st.plotly_chart(fig, key="anomaly_plots", use_container_width=True)


if __name__ == "__main__":
    main()
