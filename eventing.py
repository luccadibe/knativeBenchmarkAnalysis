import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

SAVE_TABLES = True
# Connect to the database
conn = sqlite3.connect('benchmark.db')

# Query for requests data
requests_query = """
SELECT 
    e.id as experiment_id,
    e.scenario,
    e.triggers,
    r.status,
    e.rps,
    r.ttfb,
    r.timestamp,
    r.event_id,
    e.workers
FROM experiments e
JOIN requests r ON e.id = r.experiment_id
WHERE e.scenario LIKE '%eventing%'
AND r.timestamp >= '2025-02-05T08:59:04Z'
AND r.event_id != 'None'
"""

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

events_query = """
SELECT 
    event_id,
    timestamp
FROM events
WHERE timestamp >= '2025-02-05T08:59:04Z'
"""

events_df = pd.read_sql_query(events_query, conn)
events_df['event_id'] = events_df['event_id'].astype(str)
events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ').dt.tz_localize('UTC')


requests_df = pd.read_sql_query(requests_query, conn)
requests_df['event_id'] = requests_df['event_id'].astype(str)
requests_df['timestamp'] = pd.to_datetime(requests_df['timestamp'], format='ISO8601')


node_metrics_df = pd.read_sql_query(node_metrics_query, conn)
node_metrics_df['timestamp'] = pd.to_datetime(node_metrics_df['timestamp'], format='ISO8601')
node_metrics_df['timestamp'] = node_metrics_df['timestamp'].dt.tz_convert('UTC')

pod_metrics_df = pd.read_sql_query(pod_metrics_query, conn)
pod_metrics_df['timestamp'] = pd.to_datetime(pod_metrics_df['timestamp'], format='ISO8601')
pod_metrics_df['timestamp'] = pod_metrics_df['timestamp'].dt.tz_convert('UTC')

print(f"\n Node metrics going from {node_metrics_df['timestamp'].min()} to {node_metrics_df['timestamp'].max()}")
print(f"\n Requests going from {requests_df['timestamp'].min()} to {requests_df['timestamp'].max()}")
print(f"\n Amount of requests: {requests_df.shape[0]}")

def big_plot():
    # Create the overlay plot with seaborn
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CPU utilization
    sns.lineplot(data=node_metrics_df, x='timestamp', y='cpu_percentage', 
                color='blue', ax=ax1, label='CPU Utilization')
    ax1.set_ylabel('CPU Utilization (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for TTFB
    ax2 = ax1.twinx()
    # Calculate rolling mean for TTFB to smooth the line
    requests_df['ttfb_rolling'] = requests_df['ttfb'].rolling(window=100).mean()
    sns.lineplot(data=requests_df, x='timestamp', y='ttfb_rolling', 
                color='red', ax=ax2, label='TTFB (rolling avg)')
    ax2.set_ylabel('TTFB (ms)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Improve x-axis readability
    plt.xticks(rotation=45)

    # Add title
    plt.title('CPU Utilization vs TTFB Over Time')

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adjust layout to prevent label cutoff
    fig.tight_layout()

    # Save the plot
    plt.savefig('cpu_vs_ttfb.png')
    plt.close()


def node_metrics_plot():
    # Resample data to 10 second intervals and take mean
    smoothed_df = node_metrics_df.set_index('timestamp').groupby('node_name').resample('10s')['cpu_percentage'].mean().reset_index()
    
    sns.lineplot(data=smoothed_df, x='timestamp', y='cpu_percentage', hue='node_name')
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/nodes/node_metrics_{time_now}.png')
    plt.close()


def ttfb_rolling_mean_plot():
    print("Sorting data...")
    requests_df_sorted = requests_df.sort_values('timestamp')
    
    # Calculate time difference between samples
    avg_time_diff = requests_df_sorted['timestamp'].diff().mean()
    print(f"\nAverage time between samples: {avg_time_diff}")
    
    # Use a window that represents about 1 second of data
    window_size = int(1 / avg_time_diff.total_seconds())
    print(f"Using window size of {window_size} samples (≈1 second of data)")
    
    print("Calculating rolling mean...")
    requests_df_sorted['ttfb_rolling'] = requests_df_sorted['ttfb'].rolling(
        window=window_size,
        min_periods=1
    ).mean()
    
    # Downsample to approximately 1000 points for plotting
    downsample_size = len(requests_df_sorted) // 1000
    plot_data = requests_df_sorted.iloc[::downsample_size]
    
    print(f"Plotting {len(plot_data)} points...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_data, 
        x='timestamp', 
        y='ttfb_rolling', 
        color='red'
    )
    plt.savefig('plots/ttfb_rolling_mean.png')
    plt.close()

def requests_node_metrics_plot():
    print("Preparing data...")
    # Sort and calculate rolling mean for requests data
    requests_df_sorted = requests_df.sort_values('timestamp')
    avg_time_diff = requests_df_sorted['timestamp'].diff().mean()
    window_size = int(1 / avg_time_diff.total_seconds())
    print(f"Using window size of {window_size} samples (≈1 second of data)")
    
    print("Calculating rolling mean...")
    requests_df_sorted['ttfb_rolling'] = requests_df_sorted['ttfb'].rolling(
        window=window_size,
        min_periods=1
    ).mean()
    
    # Downsample only the requests data
    requests_downsample = len(requests_df_sorted) // 1000
    plot_requests = requests_df_sorted.iloc[::requests_downsample]
    
    print(f"Plotting {len(plot_requests)} request points and {len(node_metrics_df)} metric points...")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot CPU percentage on left axis (all points)
    sns.lineplot(
        data=node_metrics_df, 
        x='timestamp', 
        y='cpu_percentage', 
        ax=ax1, 
        color='blue',
        label='CPU Usage'
    )
    
    # Plot TTFB on right axis (downsampled)
    sns.lineplot(
        data=plot_requests, 
        x='timestamp', 
        y='ttfb_rolling', 
        ax=ax2, 
        color='red',
        label='TTFB (rolling avg)'
    )
    
    # Improve labels and formatting
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='blue')
    ax2.set_ylabel('TTFB (ms)', color='red')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add title
    plt.title('CPU Usage vs TTFB Over Time')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.savefig('plots/requests_node_metrics.png')
    plt.close()
    

def process_events():
    # The basic Idea here is that the timestamp of the event in events.csv is when the event was finished processing.
    # The timestamp of the request is the time when rabbitmq sent the response that the event started processing.
    # So the difference is the time it took to process the event.
    # Calculate summary statistics for total event processing time.
    # Calculating true event processing per second is also interesting.
    # The amount of triggers is in the column triggers in the experiments table.

    # Convert timestamps to datetime objects (format: 2025-01-29T13:43:37.792661715Z)
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    requests_df['timestamp'] = pd.to_datetime(requests_df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # Merge requests and events data on event_id
    merged_df = pd.merge(
        requests_df,
        events_df,
        left_on='event_id', 
        right_on='event_id',
        suffixes=('_request', '_completion')
    )
    
    # Filter for only experiments with 1 trigger  TODO UNCOMMENT
    #merged_df = merged_df[merged_df['triggers'] == 1]

    # Calculate processing time for each event
    merged_df['processing_time'] = (
        merged_df['timestamp_completion'] - merged_df['timestamp_request']
    ).dt.total_seconds()

    # Filter out rows where processing_time is negative
    merged_df = merged_df[merged_df['processing_time'] >= 0]
    
    # Group by experiment to get summary statistics
    summary_stats = merged_df.groupby('experiment_id').agg({
        'processing_time': ['mean', 'std', 'min', 'max', 'count'],
        'triggers': 'first'  # Get number of triggers for each experiment
    }).round(3)
    
    # Calculate events processed per second
    for idx in summary_stats.index:
        experiment_data = merged_df[merged_df['rps'] == idx]
        duration = (experiment_data['timestamp_completion'].max() - 
                   experiment_data['timestamp_request'].min()).total_seconds()
        events_per_second = len(experiment_data) / duration
        summary_stats.loc[idx, ('events_per_second', '')] = round(events_per_second, 3)
    
    print("\nEvent Processing Summary Statistics:")
    print(summary_stats.to_string())
    
    # Create visualization of processing times
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged_df, x='rps', y='processing_time')
    plt.title('Event Processing Time Distribution by Experiment')
    plt.xlabel('Experiment ID')
    plt.ylabel('Processing Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/event_processing_times_{time_now}.png')
    plt.close()

    # Visualisation of knative's cpu and memory usage (per pod in knative-eventing namespace)
    print("Preparing data for visualization...")
    # Get the node where the eventing pods are running
    eventing_node = pod_metrics_df[pod_metrics_df['container_name'] == 'eventing-controller'].iloc[0]['node_name']


    # Get all of the pod metrics for the eventing node
    eventing_node_metrics = pod_metrics_df[pod_metrics_df['node_name'] == eventing_node]

    # Set the time index for resampling
    eventing_node_metrics = eventing_node_metrics.set_index('timestamp')
    merged_df_time = merged_df.set_index('timestamp_request')

    # Calculate rolling mean for event processing times
    print("Resampling event processing times...")
    plot_events = merged_df_time['processing_time'].resample('0.1S').mean()


    print(f"Plotting {len(eventing_node_metrics)} metric points and {len(plot_events)} event points...")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot the cpu usage
    sns.lineplot(data=eventing_node_metrics, x='timestamp', y='cpu_percentage',hue="pod_name", ax=ax1, color='blue')
    ax1.set_ylabel('CPU Usage (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot the event processing times
    # Plot the event processing times (resampled)
    plot_events.plot(ax=ax2, color='red', label='Event Processing Time')
    ax2.set_ylabel('Event Processing Time (seconds)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/eventing_node_metrics_{time_now}.png')
    plt.close()

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
    
    plt.savefig(f'plots/eventing/cpu_usage_{node_name}.png', bbox_inches='tight', dpi=300)
    plt.close()



def scenario_1_analysis():
    """
    Scenario 1: 
    RPS: 100, 200, 300, 400, 500, 600, 700, 800, 900
    Preliminary runs indicate that until 300rps the events are processed pretty fast.
    With 400rps we see a huge increase in latency, already 100s for some events.
    """
    print("Scenario 1 Analysis")

    # Filter for eventing scenario 1
    eventing_scenario_1_df = requests_df[requests_df['scenario'] == "eventing-scenario-1"]

    # Calculate summary statistics for TTFB grouped by rps
    summary_ttfb = eventing_scenario_1_df.groupby(
        ['rps', 'status']
    )['ttfb'].describe().round(2).reset_index()
    print("\nTTFB Summary Statistics by experiment configuration and status:")
    print(summary_ttfb.to_string(index=False))

    # Merge requests and events data on event_id
    merged_df = pd.merge(
        eventing_scenario_1_df,
        events_df,
        left_on='event_id', 
        right_on='event_id',
        suffixes=('_request', '_completion')
    )

    # Calculate processing time for each event
    merged_df['processing_time'] = (
        merged_df['timestamp_completion'] - merged_df['timestamp_request']
    ).dt.total_seconds() + (merged_df['ttfb'] / 1000)  # Convert ttfb from ms to seconds

    # Calculate summary statistics for event processing time
    summary_event_processing_time = merged_df.groupby(
        ['rps', 'status']
    )['processing_time'].describe().round(2).reset_index()
    print("\nEvent Processing Time Summary Statistics by experiment configuration and status:")
    print(summary_event_processing_time.to_string(index=False))

    # Calculate throughput (events processed per second)
    throughput_df = merged_df.copy()
    throughput_df.set_index('timestamp_completion', inplace=True)
    throughput = (throughput_df.groupby(['experiment_id', 'rps', 
                                       pd.Grouper(freq='1s')])
                 .size()
                 .reset_index(name='events_processed'))

    if SAVE_TABLES:
        summary_ttfb.to_csv('results/eventing/scenario_1_ttfb.csv', index=False)
        summary_event_processing_time.to_csv('results/eventing/scenario_1_event_processing_time_seconds.csv', index=False)
        throughput.to_csv('results/eventing/scenario_1_throughput.csv', index=False)
    plot_df = merged_df.copy()
    plot_df.set_index('timestamp_request', inplace=True)
    plot_df = plot_df.resample('1s').agg({
        'processing_time': 'mean',
        'ttfb': 'mean',
        'status': 'first',
        'rps': 'first'  # Add this line to keep the rps column
    })
    plot_df = plot_df.reset_index()

    plot_scenario_1(plot_df)

def plot_scenario_1(df):
    """
    Plot the TTFB and event processing time for scenario 1.
    """
    sns.boxplot(data=df, x='rps', y='ttfb', hue='status')
    plt.ylabel('TTFB (ms)')
    plt.savefig('plots/eventing/scenario_1_ttfb.png')
    plt.close()

    sns.boxplot(data=df, x='rps', y='processing_time', hue='status')
    plt.ylabel('Processing Time (s)') 
    plt.savefig('plots/eventing/scenario_1_processing_time_seconds.png')
    plt.close()

def scenario_2_analysis():
    """
    Scenario 2:
    Tests multiple triggers.
    """
    print("Scenario 2 Analysis")
    # Filter for eventing scenario 2
    eventing_scenario_2_df = requests_df[requests_df['scenario'] == "eventing-scenario-2"]

    # Calculate summary statistics for TTFB grouped by rps
    summary_ttfb = eventing_scenario_2_df.groupby(
        ['rps', 'status', 'triggers']
    )['ttfb'].describe().round(2).reset_index()
    print("\nTTFB in milliseconds Summary Statistics by experiment configuration and status:")
    print(summary_ttfb.to_string(index=False))


    # Merge requests and events data on event_id
    merged_df = pd.merge(
        eventing_scenario_2_df,
        events_df,
        left_on='event_id', 
        right_on='event_id',
        suffixes=('_request', '_completion')
    )

    # Calculate processing time for each event
    merged_df['processing_time'] = (
        merged_df['timestamp_completion'] - merged_df['timestamp_request']
    ).dt.total_seconds() + (merged_df['ttfb'] / 1000)  # Convert ttfb from ms to seconds

    # Calculate summary statistics for event processing time
    summary_event_processing_time = merged_df.groupby(
        ['rps', 'triggers', 'status']
    )['processing_time'].describe().round(2).reset_index()
    print("\n Scenario 2 Event Processing Time Summary Statistics by experiment configuration and status:")
    print(summary_event_processing_time.to_string(index=False))


    # Verify that each event_id has the expected number of events for each trigger configuration
    event_count_summaries = []
    
    for trigger_count in merged_df['triggers'].unique():
        # Filter data for this trigger count
        trigger_df = merged_df[merged_df['triggers'] == trigger_count]
        
        # Get unique event_ids for this trigger configuration
        unique_event_ids = trigger_df['event_id'].unique()
        total_unique_events = len(unique_event_ids)
        
        # Count events per event_id for this trigger configuration
        event_counts = trigger_df.groupby('event_id').size()
        
        # Check for mismatched counts
        mismatched_events = event_counts[event_counts != trigger_count]
        
        # Create summary for this trigger count
        summary = pd.DataFrame({
            'Triggers': [trigger_count],
            'Total Unique Events': [total_unique_events],
            'Expected Count per Event': [trigger_count],
            'Events with Expected Count': [len(event_counts[event_counts == trigger_count])],
            'Events with Mismatched Count': [len(mismatched_events)],
            'Total Event Completions': [len(trigger_df)]
        })
        
        event_count_summaries.append(summary)
        
        print(f"\nEvent Count Summary for {trigger_count} triggers:")
        print(summary.to_string(index=False))
        
        if len(mismatched_events) > 0:
            print(f"\nFound {len(mismatched_events)} events with unexpected counts")
            print(f"Expected {trigger_count} events per event_id")
            print("\nSample of events with mismatched counts:")
            print(mismatched_events.head(10))
            
            # Additional analysis of mismatched events
            print("\nDistribution of actual counts for mismatched events:")
            print(mismatched_events.value_counts().sort_index())
        else:
            print(f"\nAll events have expected count of {trigger_count} (matching number of triggers)")
    
    # Combine all summaries into one DataFrame
    event_count_summary = pd.concat(event_count_summaries, ignore_index=True)
    
    # Calculate throughput (events processed per second)
    throughput_df = merged_df.copy()
    throughput_df.set_index('timestamp_completion', inplace=True)
    throughput = (throughput_df.groupby(['experiment_id', 'rps', 
                                       pd.Grouper(freq='1s')])
                 .size()
                 .reset_index(name='events_processed'))

    # Create throughput plot
    print(f"Plotting scenario 2 throughput: {throughput.shape[0]} observations")
    plt.figure(figsize=(15, 8))
    g = sns.FacetGrid(throughput, col='rps', height=6, aspect=1.5)
    g.map(sns.lineplot, data=throughput, x='timestamp_completion', y='events_processed')
    plt.savefig('plots/eventing/scenario_2_throughput.png')
    plt.close()

    if SAVE_TABLES:
        summary_ttfb.to_csv('results/eventing/scenario_2_ttfb.csv', index=False)
        summary_event_processing_time.to_csv('results/eventing/scenario_2_event_processing_time_seconds.csv', index=False)
        throughput.to_csv('results/eventing/scenario_2_throughput.csv', index=False)
        event_count_summary.to_csv('results/eventing/scenario_2_event_count_summary.csv', index=False)
    plot_df = merged_df.copy()
    plot_df.set_index('timestamp_request', inplace=True)  # Make timestamp the index
    plot_df = plot_df.resample('1s').agg({                # Resample to 1-second intervals
        'processing_time': 'mean',  # Take average of processing times
        'ttfb': 'mean',
        'status': 'first',         # Take first status value
        'rps': 'first',            # Take first rps value
        'triggers': 'first'        # Take first triggers value
    })
    plot_df = plot_df.reset_index()

    plot_scenario_2(plot_df)

def plot_scenario_2(df):
    """
    Plot the event processing time for scenario 2.
    """
    sns.boxplot(data=df, x='rps', y='ttfb', hue='triggers')
    plt.ylabel('TTFB (ms)')
    plt.savefig('plots/eventing/scenario_2_ttfb.png')
    plt.close()

    sns.boxplot(data=df, x='rps', y='processing_time', hue='triggers')
    plt.ylabel('Processing Time (s)') 
    plt.savefig('plots/eventing/scenario_2_processing_time_seconds.png')
    plt.close()

def scenario_3_analysis():
    """
    Scenario 3:
    Variable amount of workers  and rps.
    More workers should mean less latency.
    """
    print("Scenario 3 Analysis")
    # Filter for eventing scenario 3
    eventing_scenario_3_df = requests_df[requests_df['scenario'] == "eventing-scenario-3"]

    # Calculate summary statistics for TTFB grouped by rps
    summary_ttfb = eventing_scenario_3_df.groupby(
        ['rps', 'workers', 'status']
    )['ttfb'].describe().round(2).reset_index()
    print("\nTTFB in milliseconds Summary Statistics by experiment configuration and status:")
    print(summary_ttfb.to_string(index=False))


    # Merge requests and events data on event_id
    merged_df = pd.merge(
        eventing_scenario_3_df,
        events_df,
        left_on='event_id', 
        right_on='event_id',
        suffixes=('_request', '_completion')
    )

    # Calculate processing time for each event
    merged_df['processing_time'] = (
        merged_df['timestamp_completion'] - merged_df['timestamp_request']
    ).dt.total_seconds() + (merged_df['ttfb'] / 1000)  # Convert ttfb from ms to seconds
    merged_df = merged_df[merged_df['processing_time'] >= 0]

    # Calculate summary statistics for event processing time
    summary_event_processing_time = merged_df.groupby(
        ['rps', 'workers', 'status']
    )['processing_time'].describe().round(2).reset_index()
    print("\nEvent Processing Time Summary Statistics by experiment configuration and status:")
    print(summary_event_processing_time.to_string(index=False))
    plot_scenario_3(merged_df)

    if SAVE_TABLES:
        summary_ttfb.to_csv('results/eventing/scenario_3_ttfb.csv', index=False)
        summary_event_processing_time.to_csv('results/eventing/scenario_3_event_processing_time_seconds.csv', index=False)

def plot_scenario_3(df):
    """
    Does the processing time decrease with more workers?
    """
    
    sns.boxplot(data=df, x='workers', y='processing_time', hue='rps')
    plt.savefig('plots/eventing/scenario_3_processing_time.png')
    plt.close()

    
def analyse_event_processing_time():
    """
    Analyse the event processing time. in all scenarios.
    """
    merged_df = pd.merge(
        requests_df,
        events_df,
        left_on='event_id', 
        right_on='event_id',
        suffixes=('_request', '_completion')
    )

    merged_df['processing_time'] = (
        merged_df['timestamp_completion'] - merged_df['timestamp_request']
    ).dt.total_seconds() + (merged_df['ttfb'] / 1000)  # Convert ttfb from ms to seconds
    
    # Check for negative processing times
    negative_times = merged_df[merged_df['processing_time'] < 0]
    if len(negative_times) > 0:
        print(f"\nFound {len(negative_times)} events with negative processing time")
        print("\nExample of negative processing times:")
        print(negative_times[['event_id', 'timestamp_request', 'timestamp_completion', 'processing_time']].head())
        print(f"Minimum processing time: {merged_df['processing_time'].min()}")
        print(f"Maximum processing time: {merged_df['processing_time'].max()}")
        # Filter for only negative processing times
        merged_df = merged_df[merged_df['processing_time'] <= 0]
        print(f"Average negative processing time: {merged_df['processing_time'].mean()}")
        print(f"Median negative processing time: {merged_df['processing_time'].median()}")
    else:
        print("\nNo events with negative processing time found")
    
def analyse_requests():
    """
    Event id in requests should be unique. is that so?
    """

    # Are there any event ids that are not null?
    print(f"Number of events with null event_id: {requests_df['event_id'].isna().sum()}")
    # Filter out null event IDs
    requests_filtered = requests_df[requests_df['event_id'] != "None"]
    
    # Check if event_id is unique in filtered requests
    if requests_filtered['event_id'].is_unique:
        print("\nEvent ID is unique in requests_df")
    else:
        # Count and print number of duplicates
        duplicates = requests_filtered[requests_filtered.duplicated(subset=['event_id'], keep=False)]
        print(f"\nEvent ID is not unique in requests_df")
        print(f"Found {len(duplicates)} duplicate event IDs")
        
        # Print first 100 duplicates with better formatting
        print("\nFirst 100 duplicate event IDs:")
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', None)
        pd.set_option('display.max_columns', None)
        duplicates_sample = duplicates[['event_id', 'timestamp', 'status']].head(100)
        duplicates_sample = duplicates_sample.sort_values(['event_id', 'timestamp'])
        print(duplicates_sample.to_string())
        
        # Group by event_id and show counts
        duplicate_counts = duplicates['event_id'].value_counts()
        print("\nNumber of duplicates per event ID:")
        print(duplicate_counts.head())
    
    
scenario_1_analysis()
scenario_2_analysis()
scenario_3_analysis()

conn.close()