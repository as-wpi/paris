# Walk-forward evaluation of jump-model indicators — 2026-09-02

Protocol: expanding-window selection, annual re-selection, 10 bp one-way cost, cash = T-bill. States: rolling 1,260-day calibration, monthly refit, lag 1. 'fixed' rows use library defaults throughout; '(annual sizing)' rows map states to exposure with clip(SR_k / max SR, 0, 1) estimated on the expanding history; 'selected annually' picks the best of all candidates each year.


## MKT (Fama-French)

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
MKT (Fama-French)                                                                                    
Buy & hold                          0.114  0.189   0.537 -0.546  15.221   0.748    1.000        0.000
fixed risk λ=50                     0.081  0.099   0.595 -0.219   5.490   1.474    0.497        0.763
fixed trend λ=5                     0.113  0.119   0.762 -0.203   4.919   2.303    0.666        1.812
fixed graded r50/t5                 0.098  0.096   0.778 -0.187   3.776   2.609    0.582        1.288
fixed gate r50/t5                   0.095  0.091   0.778 -0.187   3.319   2.869    0.531        1.272
fixed and r50/t5                    0.075  0.080   0.645 -0.198   3.733   2.018    0.397        0.731
fixed or r50/t5                     0.119  0.132   0.738 -0.247   6.489   1.835    0.766        1.844
cells r50/t5 (annual sizing)        0.093  0.114   0.624 -0.245   7.365   1.258    0.633        1.268
joint2f K=2 λ=5 (annual sizing)     0.105  0.137   0.622 -0.318  10.593   0.990    0.807        0.781
joint3f K=4 λ=5 (annual sizing)     0.082  0.116   0.533 -0.299   8.357   0.985    0.675        3.893
selected annually (all candidates)  0.086  0.105   0.615 -0.187   4.544   1.902    0.572        1.437
```

Annual picks: 1995: joint3f K=2 λ=5, 1996: joint3f K=4 λ=20, 1997: cells r100/t10, 1998: cells r10/t20, 1999: joint3f K=4 λ=5, 2000: joint2f K=4 λ=5, 2001: joint2f K=4 λ=5, 2002: graded r50/t20, 2003: graded r50/t20, 2004: graded r50/t20, 2005: graded r50/t20, 2006: graded r50/t20, 2007: graded r50/t20, 2008: graded r50/t20, 2009: cells r50/t10, 2010: cells r50/t5, 2011: cells r50/t5, 2012: cells r100/t10, 2013: cells r100/t10, 2014: cells r50/t5, 2015: cells r50/t10, 2016: or r50/t5, 2017: or r50/t5, 2018: graded r50/t5, 2019: graded r50/t5, 2020: graded r50/t5, 2021: graded r50/t5, 2022: graded r50/t5, 2023: or r100/t5, 2024: or r100/t5, 2025: graded r50/t5, 2026: or r100/t5


## SPY

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
SPY                                                                                                  
Buy & hold                          0.100  0.189   0.509 -0.552  12.592   0.794    1.000        0.000
fixed risk λ=50                     0.063  0.106   0.467 -0.317   7.091   0.885    0.570        0.609
fixed trend λ=5                     0.099  0.116   0.723 -0.186   5.029   1.964    0.683        1.340
fixed graded r50/t5                 0.082  0.100   0.671 -0.186   4.246   1.931    0.626        0.974
fixed gate r50/t5                   0.083  0.095   0.713 -0.186   3.588   2.320    0.574        1.096
fixed and r50/t5                    0.066  0.087   0.591 -0.186   3.410   1.945    0.466        0.853
fixed or r50/t5                     0.095  0.132   0.628 -0.317   7.886   1.206    0.787        1.096
cells r50/t5 (annual sizing)        0.074  0.091   0.652 -0.186   3.752   1.980    0.533        1.010
joint2f K=2 λ=5 (annual sizing)     0.089  0.113   0.664 -0.186   5.417   1.644    0.682        0.928
joint3f K=4 λ=5 (annual sizing)     0.049  0.106   0.348 -0.259  10.229   0.480    0.651        2.605
selected annually (all candidates)  0.065  0.108   0.488 -0.256   7.759   0.845    0.598        1.045
```

Annual picks: 2002: joint3f K=4 λ=20, 2003: cells r10/t2, 2004: cells r100/t20, 2005: cells r20/t20, 2006: cells r20/t20, 2007: cells r50/t20, 2008: cells r50/t20, 2009: and r50/t20, 2010: cells r50/t20, 2011: joint3f K=2 λ=20, 2012: joint3f K=2 λ=20, 2013: joint3f K=2 λ=20, 2014: joint3f K=2 λ=20, 2015: joint3f K=2 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint3f K=2 λ=20, 2020: joint2f K=4 λ=20, 2021: gate r50/t10, 2022: gate r50/t10, 2023: gate r50/t10, 2024: gate r50/t10, 2025: gate r50/t10, 2026: cells r100/t10


## QQQ

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
QQQ                                                                                                  
Buy & hold                          0.160  0.224   0.716 -0.500  11.992   1.340    1.000        0.000
fixed risk λ=50                     0.098  0.133   0.669 -0.228   5.489   1.787    0.550        0.697
fixed trend λ=5                     0.145  0.147   0.902 -0.195   5.531   2.628    0.663        1.770
fixed graded r50/t5                 0.123  0.129   0.859 -0.176   4.523   2.723    0.606        1.234
fixed gate r50/t5                   0.113  0.125   0.812 -0.194   4.995   2.272    0.566        1.663
fixed and r50/t5                    0.080  0.117   0.602 -0.194   6.267   1.281    0.468        1.556
fixed or r50/t5                     0.164  0.160   0.945 -0.228   5.221   3.145    0.745        0.912
cells r50/t5 (annual sizing)        0.094  0.108   0.759 -0.194   3.900   2.410    0.395        1.247
joint2f K=2 λ=5 (annual sizing)     0.143  0.153   0.863 -0.208   7.635   1.879    0.733        1.114
joint3f K=4 λ=5 (annual sizing)     0.079  0.159   0.474 -0.293   9.075   0.874    0.590        3.374
selected annually (all candidates)  0.084  0.126   0.598 -0.228   6.169   1.367    0.464        2.010
```

Annual picks: 2008: joint3f K=4 λ=5, 2009: joint3f K=4 λ=20, 2010: cells r100/t5, 2011: cells r100/t5, 2012: cells r100/t5, 2013: joint3f K=4 λ=5, 2014: cells r100/t5, 2015: cells r50/t2, 2016: cells r100/t5, 2017: cells r50/t10, 2018: cells r50/t10, 2019: cells r100/t5, 2020: cells r100/t5, 2021: or r50/t5, 2022: cells r20/t5, 2023: or r50/t5, 2024: cells r100/t5, 2025: or r50/t5, 2026: cells r50/t5


## IWM

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
IWM                                                                                                  
Buy & hold                          0.121  0.231   0.555 -0.411  11.523   1.054    1.000        0.000
fixed risk λ=50                     0.052  0.167   0.309 -0.494  23.467   0.223    0.665        0.851
fixed trend λ=5                     0.079  0.147   0.501 -0.289   7.338   1.079    0.529        1.418
fixed graded r50/t5                 0.068  0.141   0.446 -0.297   9.582   0.714    0.597        1.077
fixed gate r50/t5                   0.062  0.128   0.427 -0.245   8.050   0.766    0.466        1.361
fixed and r50/t5                    0.043  0.121   0.295 -0.274  12.790   0.334    0.404        1.304
fixed or r50/t5                     0.089  0.186   0.481 -0.421  10.834   0.824    0.790        0.851
cells r50/t5 (annual sizing)        0.038  0.118   0.263 -0.364   9.230   0.413    0.410        0.951
joint2f K=2 λ=5 (annual sizing)     0.084  0.199   0.439 -0.401  12.002   0.703    0.879        0.642
joint3f K=4 λ=5 (annual sizing)     0.073  0.126   0.519 -0.213   6.521   1.124    0.462        3.320
selected annually (all candidates)  0.054  0.121   0.389 -0.296   7.876   0.692    0.391        1.818
```

Annual picks: 2009: cells r10/t5, 2010: joint2f K=4 λ=20, 2011: joint3f K=4 λ=20, 2012: joint2f K=4 λ=20, 2013: joint2f K=4 λ=20, 2014: joint2f K=4 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=20, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=20, 2022: joint2f K=4 λ=20, 2023: joint3f K=4 λ=5, 2024: joint3f K=4 λ=5, 2025: joint3f K=4 λ=5, 2026: joint3f K=4 λ=5


## EFA

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
EFA                                                                                                  
Buy & hold                          0.072  0.183   0.394 -0.342   9.724   0.740    1.000        0.000
fixed risk λ=50                     0.013  0.154   0.068 -0.505  26.634   0.048    0.711        0.962
fixed trend λ=5                     0.054  0.140   0.345 -0.259   9.727   0.555    0.668        0.962
fixed graded r50/t5                 0.035  0.136   0.216 -0.350  17.510   0.199    0.690        0.962
fixed gate r50/t5                   0.047  0.127   0.313 -0.312  10.091   0.466    0.596        0.962
fixed and r50/t5                    0.039  0.123   0.257 -0.372  12.035   0.323    0.525        0.962
fixed or r50/t5                     0.027  0.168   0.161 -0.512  27.049   0.101    0.854        0.962
cells r50/t5 (annual sizing)        0.020  0.124   0.107 -0.386  17.419   0.114    0.547        1.270
joint2f K=2 λ=5 (annual sizing)     0.034  0.128   0.218 -0.298  15.003   0.230    0.642        0.949
joint3f K=4 λ=5 (annual sizing)     0.028  0.114   0.172 -0.355  18.090   0.153    0.392        3.682
selected annually (all candidates)  0.029  0.109   0.188 -0.282  10.210   0.285    0.405        2.954
```

Annual picks: 2010: joint3f K=4 λ=5, 2011: joint3f K=4 λ=5, 2012: cells r100/t2, 2013: joint3f K=4 λ=5, 2014: joint2f K=4 λ=5, 2015: joint3f K=4 λ=5, 2016: joint2f K=4 λ=5, 2017: joint2f K=4 λ=5, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=5, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=5, 2022: joint2f K=4 λ=5, 2023: joint2f K=4 λ=5, 2024: joint2f K=4 λ=5, 2025: joint2f K=4 λ=5, 2026: joint3f K=4 λ=20


## EEM

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
EEM                                                                                                  
Buy & hold                          0.062  0.204   0.319 -0.398  17.028   0.366    1.000        0.000
fixed risk λ=50                     0.027  0.170   0.145 -0.427  18.435   0.144    0.780        0.752
fixed trend λ=5                     0.052  0.130   0.328 -0.298  13.381   0.386    0.508        2.528
fixed graded r50/t5                 0.042  0.131   0.256 -0.281  13.496   0.311    0.644        1.640
fixed gate r50/t5                   0.045  0.114   0.305 -0.208  11.417   0.397    0.447        2.494
fixed and r50/t5                    0.037  0.108   0.246 -0.208  10.711   0.350    0.386        2.460
fixed or r50/t5                     0.041  0.185   0.221 -0.470  22.404   0.181    0.903        0.820
cells r50/t5 (annual sizing)        0.034  0.100   0.224 -0.189   4.656   0.732    0.282        1.221
joint2f K=2 λ=5 (annual sizing)     0.047  0.157   0.272 -0.344  11.537   0.412    0.622        0.696
joint3f K=4 λ=5 (annual sizing)     0.046  0.117   0.308 -0.303  11.724   0.395    0.295        3.337
selected annually (all candidates)  0.048  0.115   0.327 -0.201   4.701   1.024    0.267        1.273
```

Annual picks: 2012: cells r10/t10, 2013: cells r10/t2, 2014: cells r10/t2, 2015: cells r10/t2, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: cells r10/t10, 2019: cells r10/t20, 2020: cells r10/t20, 2021: cells r100/t2, 2022: cells r100/t2, 2023: cells r100/t2, 2024: cells r100/t2, 2025: cells r100/t2, 2026: cells r100/t10


## TLT

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
TLT                                                                                                  
Buy & hold                          0.020  0.148   0.103 -0.484  22.546   0.087    1.000        0.000
fixed risk λ=50                     0.017  0.097   0.063 -0.211  10.115   0.164    0.618        0.704
fixed trend λ=5                     0.030  0.101   0.191 -0.207  11.803   0.252    0.409        1.535
fixed graded r50/t5                 0.025  0.082   0.155 -0.160   9.014   0.274    0.514        1.119
fixed gate r50/t5                   0.015  0.072   0.037 -0.207  10.800   0.141    0.325        1.567
fixed and r50/t5                   -0.001  0.060  -0.233 -0.207  12.731  -0.005    0.241        1.599
fixed or r50/t5                     0.047  0.126   0.312 -0.213  10.368   0.459    0.787        0.640
cells r50/t5 (annual sizing)       -0.001  0.078  -0.170 -0.364  15.432  -0.008    0.320        1.456
joint2f K=2 λ=5 (annual sizing)     0.009  0.129   0.020 -0.472  21.079   0.044    0.721        0.631
joint3f K=4 λ=5 (annual sizing)     0.012  0.064  -0.024 -0.198  10.737   0.108    0.285        2.300
selected annually (all candidates)  0.042  0.081   0.368 -0.145   6.240   0.679    0.362        1.665
```

Annual picks: 2011: joint3f K=4 λ=20, 2012: cells r10/t2, 2013: cells r20/t2, 2014: cells r20/t2, 2015: cells r20/t2, 2016: cells r50/t2, 2017: cells r20/t20, 2018: cells r20/t20, 2019: cells r20/t20, 2020: cells r20/t20, 2021: cells r20/t20, 2022: cells r20/t20, 2023: cells r20/t20, 2024: cells r20/t20, 2025: cells r20/t20, 2026: cells r20/t20


## GLD

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
GLD                                                                                                  
Buy & hold                          0.069  0.166   0.382 -0.386  19.789   0.349    1.000        0.000
fixed risk λ=50                     0.038  0.113   0.238 -0.385  19.807   0.194    0.571        0.953
fixed trend λ=5                     0.057  0.128   0.362 -0.245  11.906   0.478    0.529        1.979
fixed graded r50/t5                 0.049  0.104   0.351 -0.253  12.724   0.390    0.550        1.466
fixed gate r50/t5                   0.050  0.098   0.369 -0.179   8.074   0.615    0.431        1.942
fixed and r50/t5                    0.040  0.085   0.300 -0.179   8.139   0.491    0.333        1.906
fixed or r50/t5                     0.055  0.148   0.322 -0.385  20.885   0.265    0.766        1.026
cells r50/t5 (annual sizing)        0.015  0.111   0.035 -0.358  22.215   0.068    0.516        2.346
joint2f K=2 λ=5 (annual sizing)     0.019  0.121   0.074 -0.354  22.294   0.086    0.575        1.158
joint3f K=4 λ=5 (annual sizing)     0.046  0.108   0.313 -0.256  14.037   0.330    0.333        4.174
selected annually (all candidates)  0.034  0.080   0.239 -0.196   9.985   0.338    0.265        1.597
```

Annual picks: 2013: cells r20/t5, 2014: joint2f K=4 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=20, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=5, 2022: joint2f K=4 λ=5, 2023: joint2f K=4 λ=5, 2024: joint2f K=4 λ=5, 2025: joint2f K=4 λ=5, 2026: joint2f K=4 λ=5


## XLK

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
XLK                                                                                                  
Buy & hold                          0.165  0.234   0.707 -0.531  12.989   1.276    1.000        0.000
fixed risk λ=50                     0.097  0.140   0.625 -0.257   6.205   1.568    0.558        0.560
fixed trend λ=5                     0.125  0.162   0.714 -0.292   7.583   1.646    0.694        2.037
fixed graded r50/t5                 0.113  0.138   0.736 -0.198   5.893   1.919    0.626        1.299
fixed gate r50/t5                   0.105  0.134   0.699 -0.221   6.440   1.629    0.584        1.604
fixed and r50/t5                    0.082  0.123   0.580 -0.221   7.105   1.160    0.473        1.171
fixed or r50/t5                     0.140  0.174   0.752 -0.292   6.933   2.023    0.779        1.426
cells r50/t5 (annual sizing)        0.081  0.127   0.557 -0.211   7.021   1.152    0.495        1.444
joint2f K=2 λ=5 (annual sizing)     0.137  0.171   0.748 -0.268   8.313   1.650    0.775        1.296
joint3f K=4 λ=5 (annual sizing)     0.102  0.120   0.740 -0.184   6.358   1.602    0.473        4.231
selected annually (all candidates)  0.049  0.125   0.324 -0.270   9.531   0.516    0.385        2.205
```

Annual picks: 2007: risk λ=10, 2008: joint3f K=4 λ=5, 2009: joint2f K=4 λ=20, 2010: cells r100/t10, 2011: cells r100/t10, 2012: cells r100/t10, 2013: joint2f K=4 λ=20, 2014: joint2f K=4 λ=20, 2015: cells r50/t2, 2016: joint2f K=4 λ=20, 2017: cells r100/t10, 2018: joint2f K=4 λ=20, 2019: cells r50/t10, 2020: cells r50/t10, 2021: cells r50/t2, 2022: cells r20/t2, 2023: cells r50/t2, 2024: cells r50/t2, 2025: cells r50/t2, 2026: joint3f K=4 λ=5
