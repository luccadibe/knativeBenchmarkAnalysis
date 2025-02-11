
 Node metrics going from 2025-02-05 07:39:31.552000+00:00 to 2025-02-05 15:18:58.358000+00:00

 Requests going from 2025-02-05 14:05:22.380000+00:00 to 2025-02-05 19:24:43.187000+00:00

 Amount of requests: 5405019
Scenario 1 Analysis

TTFB Summary Statistics by experiment configuration and status:
  rps  status     count   mean     std  min  25%  50%   75%     max
10000     202 1090982.0  34.25  195.17  1.0  2.0  3.0   5.0  2770.0
15000     202  581450.0 587.68 1170.07  1.0  9.0 47.0 725.0 22226.0

Event Processing Time Summary Statistics by experiment configuration and status:
  rps  status    count  mean   std   min   25%   50%   75%    max
10000     202 281256.0 50.61 30.93  0.00 24.34 51.72 80.92 107.98
15000     202 122624.0 88.04 12.80 60.04 79.20 91.28 97.46 113.38


Scenario 2 Analysis

TTFB in milliseconds Summary Statistics by experiment configuration and status:
 rps  status  triggers    count   mean    std  min  25%  50%  75%    max
3000     202         4 355787.0  18.22  62.42  1.0  2.0  3.0  4.0  889.0
4000     202         4 400215.0 145.87 616.49  1.0  2.0  3.0  5.0 6720.0
4000     202         6 814417.0  18.31  68.46  1.0  2.0  3.0  4.0  824.0
4000     202         8 798217.0  89.75 369.75  1.0  3.0  4.0  7.0 2804.0
4000     202        10 780456.0  84.88 290.09  2.0  3.0  4.0  8.0 2350.0

Event Processing Time Summary Statistics by experiment configuration and status:
 rps  triggers  status    count   mean   std    min    25%    50%    75%    max
3000         4     202 294035.0 178.48 82.54  50.23 105.28 166.75 258.41 317.29
4000         6     202 166187.0 239.16 65.40 118.27 187.36 243.14 299.40 356.94
4000         8     202 301132.0 222.82 70.36  91.10 167.64 228.61 286.91 347.37
4000        10     202 271378.0 237.86 65.22 124.29 185.46 242.34 298.74 356.61

Event Count Summary for 10 triggers:
 Triggers  Total Events  Expected Count per Event  Events with Expected Count  Events with Mismatched Count
       10         29621                        10                       25680                          3941

Found 3941 events with unexpected counts
Expected 10 events per event_id

Event Count Summary for 4 triggers:
 Triggers  Total Events  Expected Count per Event  Events with Expected Count  Events with Mismatched Count
        4         74291                         4                       72510                          1781

Found 1781 events with unexpected counts
Expected 4 events per event_id

Event Count Summary for 6 triggers:
 Triggers  Total Events  Expected Count per Event  Events with Expected Count  Events with Mismatched Count
        6         31538                         6                       23884                          7654

Found 7654 events with unexpected counts
Expected 6 events per event_id

Event Count Summary for 8 triggers:
 Triggers  Total Events  Expected Count per Event  Events with Expected Count  Events with Mismatched Count
        8         39490                         8                       35513                          3977

Found 3977 events with unexpected counts
Expected 8 events per event_id

Plotting scenario 2 throughput: 1104 observations
Scenario 3 Analysis

TTFB in milliseconds Summary Statistics by experiment configuration and status:
  rps  workers  status    count    mean     std  min   25%    50%    75%     max
10000        5     202 335476.0    2.97    1.42  1.0   2.0    3.0    3.0    42.0
15000        5     202 157161.0  744.46 1195.06  2.0  10.0   70.0  961.0  8267.0
15000       15     202  83336.0  956.16 1339.73  1.0  32.0  368.0 1533.0 17194.0
15000       20     202   7522.0 1181.84  847.15  3.0 267.0 1306.0 2054.0  2495.0

Event Processing Time Summary Statistics by experiment configuration and status:
  rps  workers  status    count  mean   std  min   25%   50%   75%   max
10000        5     202 335476.0 16.45  6.62 0.43 15.59 18.85 20.88 22.62
15000        5     202 156558.0  2.35  1.61 0.00  1.09  2.15  3.28  7.91
15000       15     202  83012.0  8.14 12.34 0.00  0.78  2.35  7.07 43.45
15000       20     202   7534.0  2.28  0.53 1.53  1.73  2.26  2.80  3.38



# Knative Eventing

The rabbitMQ broker responds with a 202 status code when an event is recieved, and it is processed asynchronously.
The event is forwarded to the trigger, which delivers it to the subscriber.
We defined the subscriber as a knative service that logs the time and id of any recieved event and forwards it to a kubernetes deployment that persists this data to a csv file.
Triggers can be configured to guarantee sequential delivery of events to the subscriber (this is the default behaviour).
Otherwise, events can be delivered in parallel by setting its "parallelism" to any value > 1.

The same challenges regarding the number of RPS to send happened with eventing as well. However, due to the rabbitMQ broker being able to buffer events more efficiently, we were able to test higher RPS values.
We defined 3 scenarios:

## Scenario 1
Sequential and parallel processing of events with a single trigger.
This single trigger was configured to deliver events in parallel with a parallelism of 10.
[![Scenario 1](results/eventing/scenario_1_ttfb.png)](results/eventing/scenario_1_ttfb.png)
TTFB Summary Statistics by experiment configuration and status:
  rps  status     count   mean     std  min  25%  50%   75%     max
10000     202 1090982.0  34.25  195.17  1.0  2.0  3.0   5.0  2770.0
15000     202  581450.0 587.68 1170.07  1.0  9.0 47.0 725.0 22226.0


[![Scenario 1](results/eventing/scenario_1_processing_time_seconds.png)](results/eventing/scenario_1_processing_time_seconds.png)
Event Processing Time Summary Statistics by experiment configuration and status:
  rps  status    count  mean   std   min   25%   50%   75%    max

10000     202 281256.0 50.61 30.93  0.00 24.34 51.72 80.92 107.98
15000     202 122624.0 88.04 12.80 60.04 79.20 91.28 97.46 113.38


## Scenario 2
Parallel processing of events with multiple triggers.
[![Scenario 2](results/eventing/scenario_2_ttfb.png)](results/eventing/scenario_2_ttfb.png)
Overall we see that the event processing time goes up with the number of triggers.
The stress on the RabbitMQ broker is much higher.

Very high variance, most responses are sent quickly (median around 4ms) but the average is 85ms for 4000 rps and 10 triggers.
rps,status,triggers,count,mean,std,min,25%,50%,75%,max
3000,202,4,355787.0,18.22,62.42,1.0,2.0,3.0,4.0,889.0
4000,202,4,400215.0,145.87,616.49,1.0,2.0,3.0,5.0,6720.0
4000,202,6,814417.0,18.31,68.46,1.0,2.0,3.0,4.0,824.0
4000,202,8,798217.0,89.75,369.75,1.0,3.0,4.0,7.0,2804.0
4000,202,10,780456.0,84.88,290.09,2.0,3.0,4.0,8.0,2350.0

The actual processing time of the events is orders of magnitude higher than the ttfb.
All units in seconds:
[![Scenario 2](results/eventing/scenario_2_processing_time_seconds.png)](results/eventing/scenario_2_processing_time_seconds.png)
rps,triggers,status,count,mean,std,min,25%,50%,75%,max

3000,4,202,294035.0,178.48,82.54,50.23,105.28,166.75,258.41,317.29
4000,6,202,166187.0,239.16,65.4,118.27,187.36,243.14,299.4,356.94
4000,8,202,301132.0,222.82,70.36,91.1,167.64,228.61,286.91,347.37
4000,10,202,271378.0,237.86,65.22,124.29,185.46,242.34,298.74,356.61

This suggests that events can be buffered in the rabbitMQ broker and delivered even many minutes after they were sent, as the load reduces.

## Scenario 3
Parallel processing of events with a single trigger and variable parallelism.
There is overall too much variance in the processing time to draw any meaningful conclusions.
[![Scenario 3](results/eventing/scenario_3_processing_time_seconds.png)](results/eventing/scenario_3_processing_time_seconds.png)
event processing time in seconds (there are some zeros due to rounding and possible sublte timing issues)
rps,workers,status,count,mean,std,min,25%,50%,75%,max


10000,5,202,335476.0,16.45,6.62,0.43,15.59,18.85,20.88,22.62
15000,5,202,156558.0,2.35,1.61,0.0,1.09,2.15,3.28,7.91
15000,15,202,83012.0,8.14,12.34,0.0,0.78,2.35,7.07,43.45
15000,20,202,7534.0,2.28,0.53,1.53,1.73,2.26,2.8,3.38

TTFB in ms of the rabbitmq broker:

rps,workers,status,count,mean,std,min,25%,50%,75%,max
10000,5,202,335476.0,2.97,1.42,1.0,2.0,3.0,3.0,42.0
15000,5,202,157161.0,744.46,1195.06,2.0,10.0,70.0,961.0,8267.0
15000,15,202,83336.0,956.16,1339.73,1.0,32.0,368.0,1533.0,17194.0
15000,20,202,7522.0,1181.84,847.15,3.0,267.0,1306.0,2054.0,2495.0