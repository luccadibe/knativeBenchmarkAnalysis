# This script is used to query a local prometheus instance for metrics regarding the knative benchmark

#autoscaler_desired_pods
#autoscaler_requested_pods
#autoscaler_terminating_pods
#autoscaler_target_pods
#autoscaler_pending_pods
#autoscaler_not_ready_pods
#autoscaler_actual_pods
#autoscaler_pod_usage
#autoscaler_pod_usage_percentage
#autoscaler_pod_usage_percentage_max
#autoscaler_pod_usage_percentage_min


#activator_request_latencies_bucket

#controller_client_latency_bucket
# The latency of the k8s api requests

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

ADDRESS = "http://localhost:9090"
PREFIX = "results/serving/prom"

# Create output directory if it doesn't exist
os.makedirs(PREFIX, exist_ok=True)

def query_prometheus(query, start_time, end_time, step='5s'):
    """Query Prometheus and return results as a DataFrame by chunking the requests"""
    # Calculate chunk size - let's use 6 hour chunks to stay well under the 11,000 point limit
    chunk_size = 6 * 3600  # 6 hours in seconds
    
    all_results = []
    current_start = start_time
    
    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        
        response = requests.get(f'{ADDRESS}/api/v1/query_range', params={
            'query': query,
            'start': current_start,
            'end': current_end,
            'step': step
        })
        
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        
        chunk_results = response.json()['data']['result']
        
        # Merge results maintaining the same structure
        if not all_results:
            all_results = chunk_results
        else:
            # Append values to existing metrics
            for i, result in enumerate(chunk_results):
                if i < len(all_results):
                    all_results[i]['values'].extend(result['values'])
        
        current_start = current_end
    
    return all_results

def extract_language(revision_name):
    """Extract language from revision name"""
    lang_mapping = {
        'quarkus': 'quarkus',
        'python': 'python',
        'go': 'go',
        'rust': 'rust',
        'springboot': 'springboot',
        'typescript': 'ts',
        'ts': 'ts'
    }
    
    for lang in lang_mapping:
        if lang in revision_name.lower():
            return lang_mapping[lang]
    return 'unknown'

def analyze_scenario_1():
    """Analyze warm latency under heavy load by language"""
    pod_state_metrics = [
        'autoscaler_desired_pods',
        'autoscaler_actual_pods',
        'autoscaler_target_pods',
        'autoscaler_requested_pods',
        'autoscaler_pending_pods',
        'autoscaler_not_ready_pods',
        'autoscaler_terminating_pods',

    ]
    
    usage_metrics = [
        'autoscaler_pod_usage_percentage'
    ]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    # Process pod state metrics together
    pod_state_data = []
    for metric in pod_state_metrics:
        results = query_prometheus(
            metric, 
            int(start_time.timestamp()), 
            int(end_time.timestamp())
        )
        
        for result in results:
            lang = extract_language(result['metric'].get('revision_name', ''))
            if lang != 'unknown':
                for timestamp, value in result['values']:
                    pod_state_data.append({
                        'timestamp': timestamp,
                        'language': lang,
                        'metric': metric,
                        'value': float(value)
                    })
    
    if pod_state_data:
        # Convert to DataFrame and pivot to get metrics as columns
        df_pod_states = pd.DataFrame(pod_state_data)
        df_pod_states_pivot = df_pod_states.pivot_table(
            index=['timestamp', 'language'],
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_pod_states_pivot.to_csv(f'{PREFIX}/scenario1_pod_states.csv', index=False)
    
    # Process usage metrics separately
    for metric in usage_metrics:
        results = query_prometheus(
            metric, 
            int(start_time.timestamp()), 
            int(end_time.timestamp())
        )
        
        usage_data = []
        for result in results:
            lang = extract_language(result['metric'].get('revision_name', ''))
            if lang != 'unknown':
                for timestamp, value in result['values']:
                    usage_data.append({
                        'timestamp': timestamp,
                        'language': lang,
                        'value': float(value)
                    })
        
        if usage_data:
            df_usage = pd.DataFrame(usage_data)
            df_usage.to_csv(f'{PREFIX}/scenario1_pod_usage.csv', index=False)

def analyze_scenario_2():
    """Analyze cold start latency by language"""
    metrics = [
        'autoscaler_not_ready_pods',
        'autoscaler_current_pods'
    ]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    cold_start_data = []
    for metric in metrics:
        results = query_prometheus(
            metric, 
            int(start_time.timestamp()), 
            int(end_time.timestamp())
        )
        
        for result in results:
            lang = extract_language(result['metric'].get('revision_name', ''))
            if lang != 'unknown':
                for timestamp, value in result['values']:
                    cold_start_data.append({
                        'timestamp': timestamp,
                        'language': lang,
                        'metric': metric,
                        'value': float(value)
                    })
    
    if cold_start_data:
        # Convert to DataFrame and pivot to get metrics as columns
        df = pd.DataFrame(cold_start_data)
        df_pivot = df.pivot_table(
            index=['timestamp', 'language'],
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_pivot.to_csv(f'{PREFIX}/scenario2_cold_start.csv', index=False)

def analyze_scenario_3():
    """Analyze impact of container concurrency=1 (Go only)"""
    metrics = [
        'autoscaler_desired_pods',
        'autoscaler_actual_pods',
        'autoscaler_target_pods',
        'autoscaler_pending_pods',
        'autoscaler_not_ready_pods',
        'autoscaler_terminating_pods',
        'autoscaler_pod_usage_percentage',
        'autoscaler_requested_pods'

    ]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    go_metrics_data = []
    for metric in metrics:
        results = query_prometheus(
            f'{metric}{{revision_name=~".*sleep.*"}}',
            int(start_time.timestamp()),
            int(end_time.timestamp())
        )
        
        for result in results:
            for timestamp, value in result['values']:
                go_metrics_data.append({
                    'timestamp': timestamp,
                    'metric': metric,
                    'value': float(value)
                })
    
    if go_metrics_data:
        # Convert to DataFrame and pivot to get metrics as columns
        df = pd.DataFrame(go_metrics_data)
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='metric',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_pivot.to_csv(f'{PREFIX}/scenario3_go_metrics.csv', index=False)

if __name__ == "__main__":
    analyze_scenario_1()
    analyze_scenario_2()
    analyze_scenario_3()
