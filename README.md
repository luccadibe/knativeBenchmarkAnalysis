This repository contains the analysis code for the Knative Serving and Eventing benchmarks.
Please see Analysis.md for the analysis of results gathered from the benchmark.

# Running the scripts
To run the scripts, use your favourite dependency manager to install the dependencies.
If you are like me and use uv, you can do the following:
```bash
uv sync
```

## With 02-09benchmark.db
```bash
uv run prometheus.py # OPTIONAL, only if you want to get the data from prometheus
uv run serving.py # This will run all the scenarios
```
This can take a while to run, so all of the already processed results and plots are in the results and plots folders.
The latest, more relevant results are always prefixed with 02-09.

## With 02-09coldstarts.db
To get the cold start data, you need to change the db name in the serving.py script.
You will also probably need to comment out some of the rest of the scenarios, as they take a long time to run.
```bash
uv run serving.py
```

## With new data
You can run the prometheus.py script to get the data from prometheus, for this you need to have a local instance of prometheus running.
You can follow the instructions in the benchmark README.md in the benchmark repository to run it.

You need to first specify the name of your database file in the serving.py script.

Then, you need to specify the nodes where knative and the functions are scheduled.
You can do this by changing the NODES_WITH_KNATIVE and NODES_WITH_FUNCTIONS variables.

This query will give you the nodes where knative is scheduled:
```sql
SELECT 
    p.node_name,
    COUNT(*) AS pod_count
FROM 
    pod_metrics p
WHERE 
    (p.pod_name LIKE '%activator%' 
     OR p.pod_name LIKE '%eventing-controller%' 
     OR p.pod_name LIKE '%autoscaler%')
    --AND p.timestamp BETWEEN 'start_time' AND 'end_time'
GROUP BY 
    p.node_name;
```

This query will give you the nodes where the functions are scheduled:
```sql
SELECT 
    p.node_name,
    COUNT(*) AS pod_count
FROM 
    pod_metrics p
WHERE 
    (p.pod_name LIKE '%empty-%')
GROUP BY 
    p.node_name;
```


And then run the scripts:

```bash
uv run prometheus.py # OPTIONAL, only if you want to get the data from prometheus
uv run serving.py
```

They may take a long while to run.
The Analysis.md file contains the analysis of the results.