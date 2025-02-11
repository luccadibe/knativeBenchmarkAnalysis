import gc
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
COLORS = {
    "go": "#42b6f5",
    "python": "#e2e83c",
    "ts": "#4287f5",
    "rust": "#f08324",
    "quarkus": "#918c87",
    "springboot": "#11c261",
}
"""
sqlite> select node_name from pod_metrics where pod_name like '%activator%' group by node_name; 
node_name
nodes-europe-west1-b-6p49
nodes-europe-west1-b-fhmc
sqlite> select node_name from pod_metrics where pod_name like '%empty-go%' group by node_name;  
node_name
nodes-europe-west1-b-2kkm
nodes-europe-west1-b-jvz1
nodes-europe-west1-b-pbj6

"""
NODES_WITH_KNATIVE = ['nodes-europe-west1-b-6p49', 'nodes-europe-west1-b-fhmc']
NODES_WITH_FUNCTIONS = ['nodes-europe-west1-b-2kkm', 'nodes-europe-west1-b-jvz1', 'nodes-europe-west1-b-pbj6']

prefix = "02-09"

SAVE_TABLES = True

# Connect to the database
conn = sqlite3.connect('02-09benchmark.db')

# Query for requests data
requests_query = """
SELECT 
    e.id as experiment_id,
    e.language,
    e.scenario,
    e.concurrency,
    e.rps,
    r.status,
    r.ttfb,
    r.dns_time,
    r.timestamp,
    r.target,
    r.is_cold
FROM experiments e
JOIN requests r ON e.id = r.experiment_id
WHERE e.scenario LIKE '%serving%'
AND r.timestamp >= '2025-02-05T08:59:04Z'
"""

# Query for node metrics
node_metrics_query = """
SELECT 
    timestamp,
    node_name,
    cpu_percentage,
    memory_percentage
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

def get_requests_df(scenario,experiment_id=None):
    print(f"Getting requests for scenario: {scenario} and experiment_id: {experiment_id}")
    start_time = time.time()
    if experiment_id:
        requests_df = pd.read_sql_query(requests_query + f" AND e.id = {experiment_id}", conn)
    else:
        requests_df = pd.read_sql_query(requests_query + f" AND e.scenario = '{scenario}'", conn)
    requests_df['timestamp'] = pd.to_datetime(requests_df['timestamp'], format='ISO8601')
    end_time = time.time()
    print(f"Time taken to get requests: {end_time - start_time} seconds")
    return requests_df

# WHERE node_name = 'nodes-europe-west1-b-s5tc' 
node_metrics_df = pd.read_sql_query(node_metrics_query, conn)
# Convert timestamps to datetime with appropriate formats
node_metrics_df['timestamp'] = pd.to_datetime(node_metrics_df['timestamp'], format='ISO8601')

# Ensure all timestamps are in UTC
node_metrics_df['timestamp'] = node_metrics_df['timestamp'].dt.tz_convert('UTC') 
print(f"\n Node metrics going from {node_metrics_df['timestamp'].min()} to {node_metrics_df['timestamp'].max()}")

pod_metrics_df = pd.read_sql_query(pod_metrics_query, conn)
pod_metrics_df['timestamp'] = pd.to_datetime(pod_metrics_df['timestamp'], format='ISO8601')
pod_metrics_df['timestamp'] = pod_metrics_df['timestamp'].dt.tz_convert('UTC')


""" node_metrics_df = None
pod_metrics_df = None """


prometheus_scenario1_df = pd.read_csv('results/serving/prom/scenario1_pod_states.csv', header=0)
prometheus_scenario2_df = pd.read_csv('results/serving/prom/scenario2_cold_start.csv', header=0)
prometheus_scenario3_df = pd.read_csv('results/serving/prom/scenario3_go_metrics.csv', header=0)



def plot_cpu_usage(node_name, pod_filter_out = [], join_on_pod_name = []):
    
    plt.figure(figsize=(20, 10))
    
    # Filter the dataframe to only include pods from the specified node
    pod_metrics_df_filtered = pod_metrics_df[pod_metrics_df['node_name'] == node_name]
    
    # Filter out pods whose names contain any of the strings in pod_filter_out
    for pod in pod_filter_out:
        pod_metrics_df_filtered = pod_metrics_df_filtered[~pod_metrics_df_filtered['pod_name'].str.contains(pod)]

    # Join (rename) pods that contain any of the base names in join_on_pod_name
    for base_name in join_on_pod_name:
        # Find all pods containing the base name and rename them to just the base name
        pod_metrics_df_filtered.loc[pod_metrics_df_filtered['pod_name'].str.contains(base_name), 'pod_name'] = base_name
    
    # Plot the filtered data
    sns.lineplot(data=pod_metrics_df_filtered, x='timestamp', y='cpu_percentage', hue="pod_name")
    
    # Move legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    plt.savefig(f'plots/serving/{prefix}_cpu_usage_{node_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

def convert_k8s_cpu(cpu_str):
    """Convert CPU from nanocores (n), microcores (u), or millicores (m) to percentage of a core"""
    if isinstance(cpu_str, str):
        if cpu_str.endswith('n'):  # nanocores
            nanocores = float(cpu_str[:-1])
            return nanocores / 10_000_000  # Convert to percentage (100% = 1 core)
        elif cpu_str.endswith('u'):  # microcores
            microcores = float(cpu_str[:-1])
            return microcores / 10_000  # Convert to percentage (100% = 1 core)
        elif cpu_str.endswith('m'):  # millicores
            millicores = float(cpu_str[:-1])
            return millicores / 10  # Convert to percentage (100% = 1 core)
    return float(cpu_str)

def convert_k8s_memory(mem_str):
    """Convert memory from Ki to MB"""
    if isinstance(mem_str, str):
        if mem_str.endswith('Ki'):
            ki = float(mem_str[:-2])
            return ki / 1024  # Convert Ki to Mi
        elif mem_str.endswith('Mi'):
            return float(mem_str[:-2])
        elif mem_str.endswith('Gi'):
            gi = float(mem_str[:-2])
            return gi * 1024  # Convert Gi to Mi
    return float(mem_str)

def scenario_1_analysisV2():
    """
    Scenario 1: 
    Languages: Go, Python, Typescript, Rust, Java (Springboot) , Java (Quarkus)
    """
    # First get all experiment IDs for scenario 1
    experiment_ids_query = """
    SELECT DISTINCT id 
    FROM experiments 
    WHERE scenario = 'serving-scenario-1'
    ORDER BY id
    """
    experiment_ids = pd.read_sql_query(experiment_ids_query, conn)['id'].tolist()
    status_data = []
    latency_data = []
    throughput_data = []
    # Process each experiment separately
    for exp_id in experiment_ids:
        print(f"Processing experiment {exp_id}")
        scenario1_df = get_requests_df("serving-scenario-1", exp_id)
        
        # Get experiment time window
        start_time = scenario1_df['timestamp'].min()
        end_time = scenario1_df['timestamp'].max()

        print(f"Start time: {start_time}, End time: {end_time} , Duration: {end_time - start_time}")
        
        # Process node metrics for this time window
        exp_nodes = node_metrics_df[
            (node_metrics_df['timestamp'] >= start_time) & 
            (node_metrics_df['timestamp'] <= end_time)
        ]


        exp_nodes = exp_nodes.copy()  # Create a copy to avoid SettingWithCopyWarning
        exp_nodes['experiment_id'] = exp_id
        exp_nodes['language'] = scenario1_df['language'].iloc[0]
        exp_nodes['rps'] = scenario1_df['rps'].iloc[0]
        exp_nodes['type'] = ["functions" if node in NODES_WITH_FUNCTIONS else "knative" if node in NODES_WITH_KNATIVE else "other" for node in exp_nodes['node_name']]
        # filter out other nodes
        exp_nodes = exp_nodes[exp_nodes['type'] != 'other']
        plot_cpu_percentage(exp_nodes)
        plot_memory_percentage(exp_nodes)

        # Collect status and latency data
        status_data.append(scenario1_df.groupby(['language', 'rps', 'status']).size())
        
        successful_requests = scenario1_df[
            (scenario1_df['status'] == 200) & 
            (~scenario1_df['is_cold'])
        ]
        latency_data.append(successful_requests)

        # Throughput analysis: compute total successful requests per second
        total_successful = len(successful_requests)
        duration = (end_time - start_time).total_seconds()
        throughput = total_successful / duration if duration > 0 else None
        throughput_data.append({
            'experiment_id': exp_id,
            'language': scenario1_df['language'].iloc[0],
            'rps': scenario1_df['rps'].iloc[0],
            'total_successful_requests': total_successful,
            'duration_seconds': duration,
            'throughput': throughput
        })
        
        # Clear memory
        del scenario1_df, exp_nodes
        gc.collect()
    
    # Combine all collected data
    status_counts = pd.concat(status_data).groupby(['language', 'rps', 'status']).sum().unstack(fill_value=0)
    latency_df = pd.concat(latency_data)
    throughput_df = pd.DataFrame(throughput_data)

    throughput_summary = throughput_df.groupby(['language', 'rps'])['throughput'].agg(
        ['min', 'mean', 'median', 'max', 'std']
    ).round(2)

    summary = latency_df.groupby(['language', 'rps'])['ttfb'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']

    dns_summary = latency_df.groupby(['language', 'rps'])['dns_time'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    dns_summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']
    status_ratios = status_counts.div(status_counts.sum(axis=1), axis=0).round(4) * 100

    # Analyse throughput as total sucessful requests per second by language and rps
    # TODO

    if SAVE_TABLES:
        status_counts.to_csv(f'results/serving/{prefix}_scenario_1_status_counts.csv', index=True)
        status_ratios.to_csv(f'results/serving/{prefix}_scenario_1_status_ratios.csv', index=True)
        summary.to_csv(f'results/serving/{prefix}_scenario_1_summary.csv', index=True)
        dns_summary.to_csv(f'results/serving/{prefix}_scenario_1_dns_summary.csv', index=True)
        throughput_summary.to_csv(f'results/serving/{prefix}_scenario_1_throughput_summary.csv', index=True)
    #Plot ttfb by language and rps
    # filter for RPS = 16000
    sampled_df = latency_df.groupby('language').apply(
        lambda x: x.sample(n=min(10000, len(x)), random_state=42)
    ).reset_index(drop=True)
    # Filter for RPS = 16000
    sampled_df = sampled_df[sampled_df['rps'] == 16000]
    # Plot TTFB
    print(f"Plotting TTFB for {sampled_df.shape[0]} observations")
    sns.ecdfplot(data=sampled_df, x='ttfb', hue='language', legend=True)
    plt.xlim(0, 500)
    plt.savefig(f'plots/serving/{prefix}_scenario_1_ttfb_ecdf.png')
    plt.close()

    # Plot DNS time
    sns.ecdfplot(data=sampled_df, x='dns_time', hue='language', legend=True)
    plt.savefig(f'plots/serving/{prefix}_scenario_1_dns_time_ecdf.png')
    plt.close()

    # Plot throughput distribution by language
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=throughput_df, x='language', y='throughput')
    plt.title("Throughput (successful requests per second) by Language")
    plt.savefig(f'plots/serving/{prefix}_scenario_1_throughput_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_cpu_percentage(df):
    """Plot CPU percentage over time for each language"""
    for lang in df['language'].unique():
        plt.figure(figsize=(15, 10))
        sns.lineplot(data=df[df['language'] == lang], x='timestamp', y='cpu_percentage', hue='type', style='node_name')
        plt.savefig(f'plots/serving/{prefix}_{lang.lower()}_cpu_percentage_by_type.png')
        plt.close()

def plot_memory_percentage(df):
    """Plot memory percentage over time for each language"""
    for lang in df['language'].unique():
        plt.figure(figsize=(15, 10))
        sns.lineplot(data=df[df['language'] == lang], x='timestamp', y='memory_percentage', hue='type', style='node_name')
        plt.savefig(f'plots/serving/{prefix}_{lang.lower()}_memory_percentage_by_type.png')
        plt.close()

def scenario_1_analysis():
    """
    Scenario 1: 
    Languages: Go, Python, Typescript, Rust, Java (Springboot) , Java (Quarkus)
    """
    # Get all scenario 1 requests
    scenario1_df = get_requests_df("serving-scenario-1")

    # Add experiment start and end timestamps
    experiment_times = scenario1_df.groupby('experiment_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    experiment_times.columns = ['experiment_id', 'start_time', 'end_time']

    # Process pod metrics for resource usage analysis
    pod_resource_data = []
    
    for _, exp in experiment_times.iterrows():
        # Filter pod metrics for this experiment's time window
        exp_pods = pod_metrics_df[
            (pod_metrics_df['timestamp'] >= exp.start_time) & 
            (pod_metrics_df['timestamp'] <= exp.end_time)
        ]
        
        # Extract language from pod name and aggregate metrics
        for lang in ['go', 'python', 'ts', 'rust', 'quarkus', 'springboot']:
            lang_pods = exp_pods[exp_pods['pod_name'].str.contains(f'-{lang}-', na=False)]
            if not lang_pods.empty:
                # Group by timestamp and sum all pod resources
                resources = lang_pods.groupby('timestamp').agg({
                    'cpu_percentage': 'sum',
                    'memory_percentage': 'sum',
                    'cpu_usage': 'sum',
                    'memory_usage': 'sum'
                }).reset_index()
                
                # Calculate statistics
                stats = {
                    'experiment_id': exp.experiment_id,
                    'language': lang,
                    'avg_cpu_percentage': resources['cpu_percentage'].mean(),
                    'max_cpu_percentage': resources['cpu_percentage'].max(),
                    'avg_memory_percentage': resources['memory_percentage'].mean(),
                    'max_memory_percentage': resources['memory_percentage'].max(),
                    'avg_cpu_usage': resources['cpu_usage'].mean(),
                    'max_cpu_usage': resources['cpu_usage'].max(),
                    'avg_memory_usage': resources['memory_usage'].mean(),
                    'max_memory_usage': resources['memory_usage'].max()
                }
                pod_resource_data.append(stats)

    # Create DataFrame with resource usage statistics
    resource_stats_df = pd.DataFrame(pod_resource_data)
    
    # Add knative components resource usage
    knative_components = ['activator', 'autoscaler']
    knative_resource_data = []
    
    for _, exp in experiment_times.iterrows():
        exp_pods = pod_metrics_df[
            (pod_metrics_df['timestamp'] >= exp.start_time) & 
            (pod_metrics_df['timestamp'] <= exp.end_time)
        ]
        
        for component in knative_components:
            component_pods = exp_pods[exp_pods['pod_name'].str.contains(component, na=False)]
            if not component_pods.empty:
                resources = component_pods.groupby('timestamp').agg({
                    'cpu_percentage': 'sum',
                    'memory_percentage': 'sum',
                    'cpu_usage': 'sum',
                    'memory_usage': 'sum'
                }).reset_index()
                
                stats = {
                    'experiment_id': exp.experiment_id,
                    'component': component,
                    'avg_cpu_percentage': resources['cpu_percentage'].mean(),
                    'max_cpu_percentage': resources['cpu_percentage'].max(),
                    'avg_memory_percentage': resources['memory_percentage'].mean(),
                    'max_memory_percentage': resources['memory_percentage'].max(),
                    'avg_cpu_usage': resources['cpu_usage'].mean(),
                    'max_cpu_usage': resources['cpu_usage'].max(),
                    'avg_memory_usage': resources['memory_usage'].mean(),
                    'max_memory_usage': resources['memory_usage'].max()
                }
                knative_resource_data.append(stats)

    knative_stats_df = pd.DataFrame(knative_resource_data)

    # Original analysis - Status codes
    status_counts = scenario1_df.groupby(['language', 'rps', 'status']).size().unstack(fill_value=0)
    status_ratios = status_counts.div(status_counts.sum(axis=1), axis=0).round(4) * 100
    
    print("\nStatus Code Distribution by Language and RPS:")
    print("\nCounts:")
    print(status_counts)
    print("\nPercentages:")
    print(status_ratios)
    
    # Filter for successful requests and non-cold starts
    successful_requests = scenario1_df[
        (scenario1_df['status'] == 200) & 
        (~scenario1_df['is_cold'])
    ]
    
    # Calculate 5-number summary grouped by language and RPS
    summary = successful_requests.groupby(['language', 'rps'])['ttfb'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    
    summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']
    print("\nLatency Summary (ms) by Language and RPS (excluding cold starts):")
    print(summary)

    # Throughput analysis
    throughput_plot = (scenario1_df[scenario1_df['status'] == 200]
                      .groupby(['language', 'rps',
                      pd.Grouper(key='timestamp', freq='1s')])
                      .size()
                      .reset_index(name='requests'))
    
    # Create throughput plot
    print(f"Plotting scenario 1 throughput: {throughput_plot.shape[0]} observations")
    plt.figure(figsize=(15, 8))
    g = sns.FacetGrid(throughput_plot, col='rps', height=6, aspect=1.5)
    g.map(sns.lineplot, data=throughput_plot, x='timestamp', y='requests', 
          hue='language', palette=COLORS)
    
    g.add_legend(title='Language')
    plt.savefig(f'plots/serving/{prefix}_scenario_1_throughput.png')
    plt.close()

    # Save all statistics
    if SAVE_TABLES:
        status_counts.to_csv(f'results/serving/{prefix}_scenario_1_status_counts.csv', index=True)
        status_ratios.to_csv(f'results/serving/{prefix}_scenario_1_status_ratios.csv', index=True)
        summary.to_csv(f'results/serving/{prefix}_scenario_1_summary.csv', index=True)
        throughput_plot.to_csv(f'results/serving/{prefix}_scenario_1_throughput.csv', index=True)
        resource_stats_df.to_csv(f'results/serving/{prefix}_scenario_1_resource_stats.csv', index=False)
        knative_stats_df.to_csv(f'results/serving/{prefix}_scenario_1_knative_stats.csv', index=False)

    # Generate all plots
    plot_df = scenario1_df.copy()
    plot_df.set_index('timestamp', inplace=True)
    plot_df = plot_df.resample('1s').agg({
        'ttfb': 'mean',
        'language': 'first',
        'rps': 'first'
    })
    plot_df = plot_df.reset_index()
    
    # Generate resource usage plots
    plot_cpu_usage_by_language(scenario1_df, pod_metrics_df, experiment_times)
    plot_memory_usage_by_language(scenario1_df, pod_metrics_df, experiment_times)

def plot_cpu_usage_by_language(scenario_df, pod_metrics, experiment_times):
    """Plot CPU usage over time for each language"""
    plt.figure(figsize=(15, 10))
    
    for _, exp in experiment_times.iterrows():
        exp_pods = pod_metrics[
            (pod_metrics['timestamp'] >= exp.start_time) & 
            (pod_metrics['timestamp'] <= exp.end_time)
        ]
        
        for lang in ['go', 'python', 'ts', 'rust', 'quarkus', 'springboot']:
            lang_pods = exp_pods[exp_pods['pod_name'].str.contains(f'-{lang}-', na=False)]
            if not lang_pods.empty:
                # Group by timestamp and sum CPU usage
                cpu_usage = lang_pods.groupby('timestamp')['cpu_percentage'].sum().reset_index()
                sns.lineplot(data=cpu_usage, x='timestamp', y='cpu_percentage', 
                           label=lang, alpha=0.7)
    
    plt.title('CPU Usage by Language Over Time')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/serving/{prefix}_scenario_1_cpu_usage.png')
    plt.close()

def plot_memory_usage_by_language(scenario_df, pod_metrics, experiment_times):
    """Plot memory usage over time for each language"""
    plt.figure(figsize=(15, 10))
    
    for _, exp in experiment_times.iterrows():
        exp_pods = pod_metrics[
            (pod_metrics['timestamp'] >= exp.start_time) & 
            (pod_metrics['timestamp'] <= exp.end_time)
        ]
        
        for lang in ['go', 'python', 'ts', 'rust', 'quarkus', 'springboot']:
            lang_pods = exp_pods[exp_pods['pod_name'].str.contains(f'-{lang}-', na=False)]
            if not lang_pods.empty:
                # Group by timestamp and sum memory usage
                memory_usage = lang_pods.groupby('timestamp')['memory_percentage'].sum().reset_index()
                sns.lineplot(data=memory_usage, x='timestamp', y='memory_percentage', 
                           label=lang, alpha=0.7)
    
    plt.title('Memory Usage by Language Over Time')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (%)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/serving/{prefix}_scenario_1_memory_usage.png')
    plt.close()

def scenario_2_analysis():
    """
    Scenario 2 (Cold start analysis): 
    Languages: Go, Python, Typescript, Rust, Java (Springboot) , Java (Quarkus)
    RPS: 0.05
    """
    # Get all scenario 2 requests and filter for cold starts
    scenario2_df = get_requests_df("serving-scenario-2")
    scenario2_df = scenario2_df[scenario2_df['is_cold'] == True]

    # Correct the language column using the target column
    # example: http://empty-ts-http-0.functions.svc.cluster.local -> ts
    scenario2_df['language'] = scenario2_df['target'].str.split('-').str[-3]
    print(f"Number of languages: {scenario2_df['language'].nunique()}")
    
    # Analyze status codes
    status_counts = scenario2_df.groupby(['language', 'status']).size().unstack(fill_value=0)
    status_ratios = status_counts.div(status_counts.sum(axis=1), axis=0).round(4) * 100
    
    print("\nStatus Code Distribution by Language:")
    print("\nCounts:")
    print(status_counts)
    print("\nPercentages:")
    print(status_ratios)

    print(f" Overall mean cold start latency: {scenario2_df['ttfb'].mean()} ms")

    successful_requests = scenario2_df[scenario2_df['status'] == 200]
    # Calculate 5-number summary grouped by language
    summary = successful_requests.groupby(['language'])['ttfb'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']

    print("\nLatency Summary (ms) by Language (only cold starts):")
    print(summary)
    if SAVE_TABLES:
        summary.to_csv(f'results/serving/{prefix}_scenario_2_summary.csv', index=True)
        status_counts.to_csv(f'results/serving/{prefix}_scenario_2_status_counts.csv', index=True)

    scenario_2_plot(scenario2_df)

def scenario_2_plot(df):
    """ECDF of Cold StartTTFB for each language - Grouped in a FacetGrid"""
    print(f"Plotting scenario 2: {df.shape[0]} observations")
    sns.ecdfplot(data=df, x='ttfb', hue='language', palette=COLORS)
    plt.title('Serving scenario 2: Cold Start TTFB in ms by Language')
    plt.savefig(f'plots/serving/{prefix}_scenario_2_plot.png')
    plt.close()


    sns.histplot(
    df,
    x="ttfb", hue="language",
    multiple="stack",
    palette=COLORS,
    edgecolor=".3",
    linewidth=.5,
    log_scale=False,
    )
    plt.title('Serving scenario 2: Cold Start TTFB by Language')
    plt.savefig(f'plots/serving/{prefix}_scenario_2_plot_hist.png')
    plt.close()

def scenario_3_analysis():
    """
    Scenario 3 Container Concurrency set to 1: 
    One language - go
    RPS: 1000 , 2000 , 3000
    """
    # Get all scenario 3 requests
    scenario3_df = get_requests_df('serving-scenario-3')

    # Filter for only rps 400
    scenario3_df = scenario3_df[scenario3_df['rps'] == 400]

    # Analyze status codes
    status_counts = scenario3_df.groupby(['language', 'rps', 'status']).size().unstack(fill_value=0)
    status_ratios = status_counts.div(status_counts.sum(axis=1), axis=0).round(4) * 100

    # Filter out cold starts
    scenario3_df = scenario3_df[scenario3_df['is_cold'] == False]
    
    print("\nStatus Code Distribution by Language and RPS:")
    print("\nCounts:")
    print(status_counts)
    print("\nPercentages:")
    print(status_ratios)

    # Calculate 5-number summary grouped by language and RPS
    summary = scenario3_df.groupby(['language', 'rps', 'status'])['ttfb'].agg([
        'count',
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    summary.columns = ['count', 'Min', '25%', 'Median', '75%', 'Max', 'Std']
    dns_summary = scenario3_df.groupby(['rps'])['dns_time'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    dns_summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']
    print("\nLatency Summary (ms) by RPS:")
    print(summary)

    # Resample while maintaining the language information
    plot_df = scenario3_df.copy()
    plot_df.set_index('timestamp', inplace=True)
    # Resample numeric data (ttfb) and use first() for categorical data (language)
    plot_df = plot_df.resample('5s').agg({
        'ttfb': 'mean',
        'rps': 'first'
    })
    plot_df = plot_df.reset_index()

    if SAVE_TABLES:
        summary.to_csv(f'results/serving/{prefix}_scenario_3_summary_ms.csv', index=True)
        status_counts.to_csv(f'results/serving/{prefix}_scenario_3_status_counts.csv', index=True)
        status_ratios.to_csv(f'results/serving/{prefix}_scenario_3_status_ratios.csv', index=True)
        dns_summary.to_csv(f'results/serving/{prefix}_scenario_3_dns_summary.csv', index=True)
    scenario_3_plot(scenario3_df)

def scenario_3_cold_start_analysis():
    """
    Scenario 3 Container Concurrency set to 1: 
    Languages: Go, Python, Typescript, Rust, Java (Springboot) , Java (Quarkus)
    RPS: 1000 , 2000 , 3000
    """
    scenario3_df = get_requests_df('serving-scenario-3')
    scenario3_df = scenario3_df[scenario3_df['is_cold'] == True]
    scenario3_df = scenario3_df[scenario3_df['status'] == 200]
    scenario3_df = scenario3_df[scenario3_df['rps'] == 400]
    # Analyze latency grouped by rps

    summary = scenario3_df.groupby(['rps'])['ttfb'].agg([
        'count',
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    summary.columns = ['count', 'Min', '25%', 'Median', '75%', 'Max', 'Std']

    dns_summary = scenario3_df.groupby(['rps'])['dns_time'].agg([
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std'
    ]).round(2)
    dns_summary.columns = ['Min', '25%', 'Median', '75%', 'Max', 'Std']
    print(f"Scenario 3 Cold Start Summary:")
    if SAVE_TABLES:
        summary.to_csv(f'results/serving/{prefix}_scenario_3_cold_start_summary.csv', index=True)
        dns_summary.to_csv(f'results/serving/{prefix}_scenario_3_cold_start_dns_summary.csv', index=True)
    sns.boxplot(data=scenario3_df, x='rps', y='ttfb', hue='rps')
    plt.savefig(f'plots/serving/{prefix}_scenario_3_cold_start_boxplot.png')
    plt.close()

def scenario_3_plot(df):
    """ECDF of TTFB"""
    print(f"Plotting scenario 3: {df.shape[0]} observations")
    sns.boxplot(data=df, y='ttfb')
    plt.title('Serving scenario 3: ContainerConcurrency set to 1 - TTFB by RPS')
    plt.savefig(f'plots/serving/{prefix}_scenario_3_plot.png')
    plt.close()

scenario_1_analysisV2()
# garbage collect
gc.collect()
scenario_2_analysis()
#garbage collect
gc.collect()
scenario_3_analysis()
# garbage collect
gc.collect()
scenario_3_cold_start_analysis()
# garbage collect
gc.collect()
conn.close()
