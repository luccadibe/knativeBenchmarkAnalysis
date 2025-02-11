import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Connect to the database
conn = sqlite3.connect('02-08benchmark.db')

# Query for node metrics
node_metrics_query = """
SELECT 

    timestamp,
    node_name,
    cpu_percentage
FROM node_metrics
ORDER BY timestamp
"""

# Query for pod metrics
pod_metrics_query = """
SELECT 
    timestamp,
    container_name,
    pod_name,
    node_name,
    cpu_percentage,
    memory_percentage,
    cpu_usage,
    memory_usage
FROM pod_metrics
ORDER BY timestamp
"""
prefix = "02-08"

node_metrics_df = pd.read_sql_query(node_metrics_query, conn)
node_metrics_df['timestamp'] = pd.to_datetime(node_metrics_df['timestamp'], format='ISO8601')
node_metrics_df['timestamp'] = node_metrics_df['timestamp'].dt.tz_convert('UTC')

pod_metrics_df = pd.read_sql_query(pod_metrics_query, conn)
pod_metrics_df['timestamp'] = pd.to_datetime(pod_metrics_df['timestamp'], format='ISO8601')
pod_metrics_df['timestamp'] = pod_metrics_df['timestamp'].dt.tz_convert('UTC')


def node_metrics_plot():
    # Resample data to 10 second intervals and take mean
    smoothed_df = node_metrics_df.set_index('timestamp').groupby('node_name').resample('10s')['cpu_percentage'].mean().reset_index()
    
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=smoothed_df, x='timestamp', y='cpu_percentage', hue='node_name')
    
    # Move legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/nodes/{prefix}_node_metrics_{time_now}.png', bbox_inches='tight', dpi=300)
    plt.close()

node_metrics_plot()

conn.close()