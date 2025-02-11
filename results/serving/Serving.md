 Node metrics going from 2025-02-05 07:39:31.552000+00:00 to 2025-02-05 15:18:58.358000+00:00

 Requests going from 2025-02-05 09:52:00.282000+00:00 to 2025-02-05 13:51:33.486000+00:00

 Amount of requests: 22093887

Status Code Distribution by Language and RPS:

Counts:
status                200  408   502      503
language   rps
go         10000   673928    0     0        0
           11000  1709088    0     0    46473
           12000  1085774    0     0    87082
python     10000   261037    0     5   740990
           11000   452541    0  1203  1489055
           12000   151927    0   124   795807
quarkus    10000   173343    0     0   735914
           11000   538856    0     0  1201733
           12000    26022    0     0   897341
rust       10000   346361    4     0   741679
           11000   346011  146     0   688556
           12000        0    0     0  1192352
springboot 10000   106754    0     2   865590
           11000    58335    0    45  1151383
           12000        0    0     0  1302125
ts         10000        0    0     0  1328392
           11000   385027    0    89   551279
           12000        0    0     0  1219213

Percentages:
status               200   408   502     503
language   rps
go         10000  100.00  0.00  0.00    0.00
           11000   97.35  0.00  0.00    2.65
           12000   92.58  0.00  0.00    7.42
python     10000   26.05  0.00  0.00   73.95
           11000   23.29  0.00  0.06   76.64
           12000   16.03  0.00  0.01   83.96
quarkus    10000   19.06  0.00  0.00   80.94
           11000   30.96  0.00  0.00   69.04
           12000    2.82  0.00  0.00   97.18
rust       10000   31.83  0.00  0.00   68.17
           11000   33.44  0.01  0.00   66.55
           12000    0.00  0.00  0.00  100.00
springboot 10000   10.98  0.00  0.00   89.02
           11000    4.82  0.00  0.00   95.17
           12000    0.00  0.00  0.00  100.00
ts         10000    0.00  0.00  0.00  100.00
           11000   41.12  0.00  0.01   58.87
           12000    0.00  0.00  0.00  100.00

Latency Summary (ms) by Language and RPS (excluding cold starts):
                  Min     25%   Median       75%      Max      Std
language   rps
go         10000  0.0     0.0      0.0      1.00     55.0     0.61
           11000  0.0     1.0      1.0      9.00  29463.0  1322.09
           12000  0.0     1.0      3.0     18.00  29999.0  2819.26
python     10000  2.0  1590.0   2966.0   5525.00  30323.0  6239.53
           11000  2.0  3111.0   6343.0  12154.00  29999.0  7776.19
           12000  2.0  3254.0   7028.0  16197.00  30281.0  8363.00
rust       10000  0.0     1.0      4.0     13.00  25894.0    51.97
           11000  0.0     2.0     30.0   1686.00  29955.0  1899.63
springboot 10000  2.0  3809.0   8984.0  18206.00  30000.0  8792.19
           11000  3.0  8123.0  14920.0  21329.25  31780.0  8158.61
ts         11000  0.0   528.0   5556.0  16476.00  30022.0  9244.15
Plotting scenario 1 throughput: 2686 observations
Plotting scenario 1: 8767 observations
Number of languages: 6

Status Code Distribution by Language:

Counts:
status      200
language
go          240
python      238
quarkus     240
rust        240
springboot  239
ts          239

Percentages:
status        200
language
go          100.0
python      100.0
quarkus     100.0
rust        100.0
springboot  100.0
ts          100.0

Latency Summary (ms) by Language (only cold starts):
               Min      25%  Median      75%      Max     Std
language
go          1080.0  1698.50  1939.0  2228.25   3152.0  354.66
python      2262.0  2463.25  2535.5  2644.25   8959.0  434.16
quarkus        3.0  2813.50  2869.0  2925.25  11224.0  610.94
rust        1076.0  1761.50  2020.5  2295.50   2689.0  361.55
springboot  1126.0  1675.50  1943.0  2216.00   4209.0  372.59
ts          1512.0  1900.50  2097.0  2353.50  10239.0  593.78
Plotting scenario 2: 1436 observations

Status Code Distribution by Language and RPS:

Counts:
status            200  503
language rps
go       1000  183610    0
         2000  252684    0
         3000  304376  111

Percentages:
status            200   503
language rps
go       1000  100.00  0.00
         2000  100.00  0.00
         3000   99.96  0.04

Latency Summary (ms) by Language and RPS (excluding cold starts):
               Min  25%  Median  75%     Max     Std
language rps
go       1000  0.0  0.0     0.0  1.0    12.0    0.52
         2000  0.0  0.0     0.0  1.0    53.0    1.32
         3000  0.0  1.0     1.0  1.0  3174.0  173.50
Plotting scenario 3: 740781 observations


Scenario 3 Cold Start Summary:
      count  Min   25%  Median    75%    Max     Std
rps
1000      2  2.0  2.25     2.5   2.75    3.0    0.71
2000     10  1.0  1.00     2.0   2.00    6.0    1.52