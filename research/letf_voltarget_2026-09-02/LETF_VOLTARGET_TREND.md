# Trend gate on top of the vol-target sizing (study §14.1 construction) — 2026-09-02

w* = clip(30 % / EWMA-vol of the leveraged fund, 0, 1); band [25 %, 35 %] on implied vol; rebalance on gate flip or band exit; t+0; 8 bp one-way; T-bill cash. Gates on the 1x fund at the close of T-1. OOS from the leveraged fund's inception.


## SPY → UPRO (3x), 2009-09-23 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
SPY/UPRO 2009-09-23..2026-09-02                                                                                                 
B&H leveraged                        0.299  0.511   0.745 -0.768  21.736   1.377    1.000        0.000           NaN         NaN
VT30, no gate                        0.207    NaN   0.743 -0.340  13.668   1.520    0.729        2.787         0.294       0.729
200SMA gate + VT30 (study headline)  0.163    NaN   0.653 -0.315  14.688   1.115    0.666        4.878         0.265       0.666
trend gate + VT30                    0.166    NaN   0.691 -0.327  13.850   1.200    0.550        2.898         0.247       0.550
trend gate + VT30, no band           0.171    NaN   0.702 -0.329  13.970   1.226    0.564        4.168         0.250       0.564
trend gate, unsized (w=1)            0.254    NaN   0.768 -0.519  16.857   1.509    0.709        1.419         0.365       0.709
```


## QQQ → TQQQ (3x), 2010-05-12 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
QQQ/TQQQ 2010-05-12..2026-09-02                                                                                                 
B&H leveraged                        0.410  0.613   0.848 -0.816  27.111   1.517    1.000        0.000           NaN         NaN
VT30, no gate                        0.268    NaN   0.890 -0.425  14.202   1.894    0.601        2.872         0.303       0.601
200SMA gate + VT30 (study headline)  0.198    NaN   0.739 -0.383  16.305   1.215    0.546        5.167         0.278       0.546
trend gate + VT30                    0.227    NaN   0.874 -0.294  12.573   1.812    0.454        3.214         0.256       0.454
trend gate + VT30, no band           0.229    NaN   0.871 -0.293  12.716   1.804    0.463        4.648         0.260       0.463
trend gate, unsized (w=1)            0.357    NaN   0.874 -0.515  18.351   1.951    0.706        1.843         0.452       0.706
```


## XLK → TECL (3x), 2009-03-19 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
XLK/TECL 2009-03-19..2026-09-02                                                                                                 
B&H leveraged                        0.472  0.659   0.900 -0.780  26.236   1.804    1.000        0.000           NaN         NaN
VT30, no gate                        0.271    NaN   0.902 -0.368  13.400   2.023    0.572        2.957         0.302       0.572
200SMA gate + VT30 (study headline)  0.210    NaN   0.779 -0.348  15.991   1.313    0.520        4.722         0.277       0.520
trend gate + VT30                    0.195    NaN   0.772 -0.323  12.420   1.576    0.434        3.128         0.257       0.434
trend gate + VT30, no band           0.193    NaN   0.755 -0.320  12.937   1.495    0.446        4.296         0.262       0.446
trend gate, unsized (w=1)            0.297    NaN   0.746 -0.691  21.600   1.378    0.724        2.008         0.505       0.724
```


## IWM → TNA (3x), 2009-02-05 .. 2026-09-02

```
                                      CAGR   Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
IWM/TNA 2009-02-05..2026-09-02                                                                                                 
B&H leveraged                        0.161  0.68   0.546 -0.881  41.548   0.389    1.000        0.000           NaN         NaN
VT30, no gate                        0.098   NaN   0.417 -0.511  23.165   0.423    0.538        2.916         0.301       0.538
200SMA gate + VT30 (study headline)  0.056   NaN   0.290 -0.508  27.007   0.209    0.431        6.144         0.254       0.431
trend gate + VT30                    0.083   NaN   0.410 -0.419  16.477   0.502    0.311        2.296         0.220       0.311
trend gate + VT30, no band           0.077   NaN   0.382 -0.412  16.663   0.462    0.320        3.142         0.226       0.320
trend gate, unsized (w=1)            0.139   NaN   0.487 -0.706  25.405   0.548    0.532        1.425         0.439       0.532
```


## EEM → EDC (3x), 2009-04-17 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
EEM/EDC 2009-04-17..2026-09-02                                                                                                  
B&H leveraged                        0.009  0.654   0.324 -0.925  69.766   0.013    1.000        0.000           NaN         NaN
VT30, no gate                        0.023    NaN   0.181 -0.663  39.003   0.058    0.539        2.927         0.303       0.539
200SMA gate + VT30 (study headline)  0.009    NaN   0.100 -0.663  44.696   0.020    0.375        6.956         0.239       0.375
trend gate + VT30                    0.045    NaN   0.249 -0.478  31.212   0.145    0.316        2.824         0.227       0.316
trend gate + VT30, no band           0.041    NaN   0.231 -0.499  33.278   0.123    0.325        3.770         0.232       0.325
trend gate, unsized (w=1)            0.034    NaN   0.272 -0.795  60.776   0.056    0.555        2.248         0.452       0.555
```


## TLT → TMF (3x), 2009-07-15 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
TLT/TMF 2009-07-15..2026-09-02                                                                                                  
B&H leveraged                       -0.046  0.444   0.084 -0.931  52.024  -0.089    1.000        0.000           NaN         NaN
VT30, no gate                       -0.002    NaN   0.094 -0.807  40.895  -0.005    0.744        2.192         0.295       0.744
200SMA gate + VT30 (study headline) -0.032    NaN  -0.101 -0.777  38.272  -0.085    0.412        9.772         0.221       0.412
trend gate + VT30                    0.020    NaN   0.127 -0.557  27.766   0.072    0.302        2.069         0.192       0.302
trend gate + VT30, no band           0.019    NaN   0.124 -0.571  29.043   0.066    0.309        2.896         0.196       0.309
trend gate, unsized (w=1)            0.021    NaN   0.175 -0.660  37.535   0.056    0.414        1.520         0.305       0.414
```


## GLD → UGL (2x), 2010-11-22 .. 2026-09-02

```
                                      CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr  Realised vol  Avg weight
GLD/UGL 2010-11-22..2026-09-02                                                                                                  
B&H leveraged                        0.076  0.337   0.343 -0.759  52.400   0.145    1.000        0.000           NaN         NaN
VT30, no gate                        0.089    NaN   0.397 -0.696  44.244   0.202    0.887        1.428         0.273       0.887
200SMA gate + VT30 (study headline)  0.067    NaN   0.340 -0.574  39.890   0.169    0.544        9.140         0.218       0.544
trend gate + VT30                    0.078    NaN   0.397 -0.515  30.941   0.253    0.463        2.729         0.206       0.463
trend gate + VT30, no band           0.076    NaN   0.385 -0.524  31.798   0.239    0.470        3.582         0.207       0.470
trend gate, unsized (w=1)            0.067    NaN   0.323 -0.604  41.078   0.164    0.536        2.032         0.266       0.536
```


## Sharpe across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                           0.745     0.848     0.900    0.546    0.324    0.084    0.343  0.541
VT30, no gate                           0.743     0.890     0.902    0.417    0.181    0.094    0.397  0.518
200SMA gate + VT30 (study headline)     0.653     0.739     0.779    0.290    0.100   -0.101    0.340  0.400
trend gate + VT30                       0.691     0.874     0.772    0.410    0.249    0.127    0.397  0.503
trend gate + VT30, no band              0.702     0.871     0.755    0.382    0.231    0.124    0.385  0.493
trend gate, unsized (w=1)               0.768     0.874     0.746    0.487    0.272    0.175    0.323  0.521
```


## MaxDD across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                          -0.768    -0.816    -0.780   -0.881   -0.925   -0.931   -0.759 -0.837
VT30, no gate                          -0.340    -0.425    -0.368   -0.511   -0.663   -0.807   -0.696 -0.544
200SMA gate + VT30 (study headline)    -0.315    -0.383    -0.348   -0.508   -0.663   -0.777   -0.574 -0.510
trend gate + VT30                      -0.327    -0.294    -0.323   -0.419   -0.478   -0.557   -0.515 -0.416
trend gate + VT30, no band             -0.329    -0.293    -0.320   -0.412   -0.499   -0.571   -0.524 -0.421
trend gate, unsized (w=1)              -0.519    -0.515    -0.691   -0.706   -0.795   -0.660   -0.604 -0.641
```


## Martin across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                           1.377     1.517     1.804    0.389    0.013   -0.089    0.145  0.737
VT30, no gate                           1.520     1.894     2.023    0.423    0.058   -0.005    0.202  0.874
200SMA gate + VT30 (study headline)     1.115     1.215     1.313    0.209    0.020   -0.085    0.169  0.565
trend gate + VT30                       1.200     1.812     1.576    0.502    0.145    0.072    0.253  0.794
trend gate + VT30, no band              1.226     1.804     1.495    0.462    0.123    0.066    0.239  0.774
trend gate, unsized (w=1)               1.509     1.951     1.378    0.548    0.056    0.056    0.164  0.809
```


## CAGR across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                           0.299     0.410     0.472    0.161    0.009   -0.046    0.076  0.197
VT30, no gate                           0.207     0.268     0.271    0.098    0.023   -0.002    0.089  0.136
200SMA gate + VT30 (study headline)     0.163     0.198     0.210    0.056    0.009   -0.032    0.067  0.096
trend gate + VT30                       0.166     0.227     0.195    0.083    0.045    0.020    0.078  0.116
trend gate + VT30, no band              0.171     0.229     0.193    0.077    0.041    0.019    0.076  0.115
trend gate, unsized (w=1)               0.254     0.357     0.297    0.139    0.034    0.021    0.067  0.167
```


## Turnover/yr across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                           0.000     0.000     0.000    0.000    0.000    0.000    0.000  0.000
VT30, no gate                           2.787     2.872     2.957    2.916    2.927    2.192    1.428  2.583
200SMA gate + VT30 (study headline)     4.878     5.167     4.722    6.144    6.956    9.772    9.140  6.683
trend gate + VT30                       2.898     3.214     3.128    2.296    2.824    2.069    2.729  2.737
trend gate + VT30, no band              4.168     4.648     4.296    3.142    3.770    2.896    3.582  3.786
trend gate, unsized (w=1)               1.419     1.843     2.008    1.425    2.248    1.520    2.032  1.785
```


## Avg weight across funds

```
                                     SPY/UPRO  QQQ/TQQQ  XLK/TECL  IWM/TNA  EEM/EDC  TLT/TMF  GLD/UGL   mean
B&H leveraged                             NaN       NaN       NaN      NaN      NaN      NaN      NaN    NaN
VT30, no gate                           0.729     0.601     0.572    0.538    0.539    0.744    0.887  0.659
200SMA gate + VT30 (study headline)     0.666     0.546     0.520    0.431    0.375    0.412    0.544  0.499
trend gate + VT30                       0.550     0.454     0.434    0.311    0.316    0.302    0.463  0.404
trend gate + VT30, no band              0.564     0.463     0.446    0.320    0.325    0.309    0.470  0.414
trend gate, unsized (w=1)               0.709     0.706     0.724    0.532    0.555    0.414    0.536  0.597
```
