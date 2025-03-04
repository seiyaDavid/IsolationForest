import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_stock_data(n_rows=500, n_stocks=5):
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_rows)]
    
    # Create DataFrame with dates
    df = pd.DataFrame({'Date': dates})
    
    # Generate stock data with some anomalies
    for i in range(n_stocks):
        stock_name = f'STOCK_{i+1}'
        
        # Generate normal returns
        returns = np.random.normal(0.001, 0.02, n_rows)
        
        # Insert some anomalies (extreme values)
        n_anomalies = int(n_rows * 0.05)  # 5% anomalies
        anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
        returns[anomaly_indices] = np.random.choice(
            [np.random.uniform(-0.15, -0.1), np.random.uniform(0.1, 0.15)],
            n_anomalies
        )
        
        df[stock_name] = returns

    return df

# Create sample_data directory if it doesn't exist
os.makedirs('sample_data', exist_ok=True)

# Generate and save data
df = generate_stock_data()
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.to_csv('sample_data/stock_data.csv', index=False)

print("Sample data generated and saved to sample_data/stock_data.csv")
print("\nFirst few rows:")
print(df.head())
print("\nShape:", df.shape) 