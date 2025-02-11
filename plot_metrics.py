import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [15, 10]

# Input/Output directories
INPUT_DIR = "results/serving/prom"
OUTPUT_DIR = "results/serving/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILTER_TIMESTAMP = "2025-02-09 11:07:00"

def plot_scenario1():
    """Plot pod states metrics for each language"""
    # Read the data
    pod_states_df = pd.read_csv(f"{INPUT_DIR}/scenario1_pod_states.csv")
    
    # Convert timestamp to datetime
    pod_states_df['timestamp'] = pd.to_datetime(pod_states_df['timestamp'], unit='s')
    
    # Filter data after specified timestamp
    pod_states_df = pod_states_df[pod_states_df['timestamp'] >= FILTER_TIMESTAMP]
    
    # Get unique languages
    languages = pod_states_df['language'].unique()
    
    for lang in languages:
        plt.figure(figsize=(15, 10))
        
        # Filter data for current language
        lang_states = pod_states_df[pod_states_df['language'] == lang]
        
        # Plot pod states
        metrics_to_plot = [
            'autoscaler_actual_pods',
            'autoscaler_desired_pods',
            'autoscaler_requested_pods',
            'autoscaler_pending_pods',
            'autoscaler_not_ready_pods',
            'autoscaler_terminating_pods'
        ]
        
        for metric in metrics_to_plot:
            sns.lineplot(data=lang_states, x='timestamp', y=metric, label=metric.replace('autoscaler_', ''))
        
        plt.title(f'Pod States Over Time - {lang.upper()}')
        plt.xlabel('Time')
        plt.ylabel('Number of Pods')
        plt.ylim(0, 400)  # Set y-axis limit
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/scenario1_{lang}_metrics.png")
        plt.close()

def plot_scenario3():
    """Plot all metrics for Go concurrency test"""
    # Read the data
    df = pd.read_csv(f"{INPUT_DIR}/scenario3_go_metrics.csv")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Filter data after specified timestamp
    #df = df[df['timestamp'] >= FILTER_TIMESTAMP]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot all metrics
    metrics_to_plot = [
        'autoscaler_actual_pods',
        'autoscaler_desired_pods',
        'autoscaler_requested_pods',
        'autoscaler_pending_pods',
        'autoscaler_not_ready_pods',
        'autoscaler_terminating_pods'
    ]
    
    for metric in metrics_to_plot:
        sns.lineplot(data=df, x='timestamp', y=metric, label=metric.replace('autoscaler_', ''))
    
    plt.title('Go Concurrency Test Metrics')
    plt.xlabel('Time')
    plt.ylabel('Number of Pods')
    plt.ylim(0, 400)  # Set y-axis limit
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scenario3_metrics.png")
    plt.close()

if __name__ == "__main__":
    plot_scenario1()
    plot_scenario3() 