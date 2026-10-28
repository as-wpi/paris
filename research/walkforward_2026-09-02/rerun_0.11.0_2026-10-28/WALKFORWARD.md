# Walk-forward evaluation of jump-model indicators — 2026-10-28

Protocol: expanding-window selection, annual re-selection, 10 bp one-way cost, cash = T-bill. States: rolling 1,260-day calibration, monthly refit, lag 1. 'fixed' rows use library defaults throughout; '(annual sizing)' rows map states to exposure with clip(SR_k / max SR, 0, 1) estimated on the expanding history; 'selected annually' picks the best of all candidates each year.


## MKT (Fama-French)

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
MKT (Fama-French)                                                                                    
Buy & hold                          0.114  0.189   0.537 -0.546  15.221   0.748    1.000        0.000
fixed risk λ=50                     0.080  0.098   0.584 -0.219   5.490   1.452    0.495        0.700
fixed trend λ=5                     0.115  0.117   0.782 -0.203   4.636   2.482    0.661        1.558
fixed graded r50/t5                 0.099  0.095   0.784 -0.187   3.811   2.591    0.578        1.129
fixed gate r50/t5                   0.095  0.091   0.784 -0.187   3.276   2.913    0.528        1.113
fixed and r50/t5                    0.074  0.080   0.633 -0.198   3.733   1.986    0.394        0.668
fixed or r50/t5                     0.121  0.131   0.755 -0.247   6.527   1.852    0.762        1.590
cells r50/t5 (annual sizing)        0.095  0.114   0.642 -0.234   7.286   1.300    0.632        1.148
joint2f K=2 λ=5 (annual sizing)     0.104  0.136   0.618 -0.313  10.458   0.993    0.797        0.783
joint3f K=4 λ=5 (annual sizing)     0.088  0.119   0.567 -0.329   9.062   0.972    0.694        3.863
selected annually (all candidates)  0.088  0.112   0.599 -0.221   5.614   1.579    0.620        1.728
```

Annual picks: 1995: cells r20/t10, 1996: joint3f K=4 λ=5, 1997: cells r100/t10, 1998: cells r50/t20, 1999: joint3f K=4 λ=5, 2000: joint2f K=4 λ=5, 2001: joint2f K=4 λ=5, 2002: graded r50/t20, 2003: graded r50/t20, 2004: cells r50/t20, 2005: cells r50/t20, 2006: cells r50/t20, 2007: cells r50/t20, 2008: cells r50/t20, 2009: cells r50/t5, 2010: cells r50/t5, 2011: cells r50/t5, 2012: joint3f K=4 λ=5, 2013: cells r50/t5, 2014: joint3f K=4 λ=5, 2015: cells r50/t5, 2016: or r50/t5, 2017: or r50/t5, 2018: graded r50/t5, 2019: cells r50/t5, 2020: cells r50/t5, 2021: cells r50/t5, 2022: cells r50/t5, 2023: or r100/t5, 2024: or r100/t5, 2025: or r100/t5, 2026: cells r100/t5


## SPY

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
SPY                                                                                                  
Buy & hold                          0.100  0.189   0.509 -0.552  12.592   0.794    1.000        0.000
fixed risk λ=50                     0.062  0.106   0.465 -0.300   6.609   0.945    0.566        0.609
fixed trend λ=5                     0.095  0.117   0.693 -0.186   5.660   1.684    0.686        1.421
fixed graded r50/t5                 0.080  0.100   0.653 -0.186   4.238   1.890    0.626        1.015
fixed gate r50/t5                   0.081  0.095   0.694 -0.186   3.751   2.170    0.576        1.137
fixed and r50/t5                    0.066  0.087   0.588 -0.186   3.411   1.935    0.465        0.853
fixed or r50/t5                     0.091  0.132   0.602 -0.300   7.884   1.162    0.787        1.177
cells r50/t5 (annual sizing)        0.070  0.090   0.611 -0.186   3.970   1.759    0.523        1.048
joint2f K=2 λ=5 (annual sizing)     0.090  0.113   0.672 -0.186   5.286   1.703    0.682        0.848
joint3f K=4 λ=5 (annual sizing)     0.049  0.116   0.329 -0.281  10.352   0.478    0.661        2.597
selected annually (all candidates)  0.057  0.105   0.420 -0.300   8.496   0.672    0.585        1.283
```

Annual picks: 2002: joint3f K=4 λ=5, 2003: cells r10/t2, 2004: cells r100/t20, 2005: cells r20/t20, 2006: cells r20/t20, 2007: cells r50/t20, 2008: cells r50/t20, 2009: and r50/t20, 2010: cells r50/t20, 2011: joint3f K=2 λ=20, 2012: joint3f K=2 λ=20, 2013: joint3f K=2 λ=20, 2014: joint3f K=2 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=20, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=20, 2022: joint3f K=2 λ=20, 2023: joint3f K=2 λ=20, 2024: joint2f K=4 λ=20, 2025: joint2f K=2 λ=5, 2026: gate r50/t10


## QQQ

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
QQQ                                                                                                  
Buy & hold                          0.160  0.224   0.716 -0.500  11.992   1.340    1.000        0.000
fixed risk λ=50                     0.100  0.133   0.682 -0.228   5.079   1.969    0.551        0.805
fixed trend λ=5                     0.144  0.147   0.897 -0.195   5.505   2.625    0.661        1.663
fixed graded r50/t5                 0.124  0.129   0.863 -0.176   4.521   2.737    0.606        1.234
fixed gate r50/t5                   0.113  0.125   0.813 -0.194   4.941   2.300    0.564        1.610
fixed and r50/t5                    0.081  0.117   0.611 -0.194   5.611   1.451    0.467        1.556
fixed or r50/t5                     0.164  0.160   0.945 -0.228   5.221   3.145    0.745        0.912
cells r50/t5 (annual sizing)        0.094  0.110   0.747 -0.194   3.931   2.387    0.402        1.212
joint2f K=2 λ=5 (annual sizing)     0.144  0.153   0.867 -0.201   7.529   1.911    0.732        1.186
joint3f K=4 λ=5 (annual sizing)     0.086  0.135   0.577 -0.288   8.101   1.060    0.538        3.072
selected annually (all candidates)  0.081  0.123   0.583 -0.260   6.677   1.215    0.454        1.972
```

Annual picks: 2008: joint3f K=4 λ=5, 2009: joint3f K=4 λ=20, 2010: cells r100/t5, 2011: cells r100/t5, 2012: cells r100/t5, 2013: cells r100/t5, 2014: cells r100/t5, 2015: cells r50/t2, 2016: cells r100/t5, 2017: cells r100/t5, 2018: cells r50/t5, 2019: cells r100/t5, 2020: cells r100/t5, 2021: or r50/t5, 2022: cells r20/t5, 2023: or r50/t5, 2024: cells r100/t2, 2025: or r50/t5, 2026: cells r50/t5


## IWM

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
IWM                                                                                                  
Buy & hold                          0.121  0.231   0.555 -0.411  11.523   1.054    1.000        0.000
fixed risk λ=50                     0.054  0.167   0.318 -0.494  23.467   0.229    0.669        0.737
fixed trend λ=5                     0.081  0.148   0.511 -0.289   7.323   1.107    0.530        1.644
fixed graded r50/t5                 0.070  0.141   0.457 -0.289   9.155   0.766    0.600        1.134
fixed gate r50/t5                   0.063  0.129   0.437 -0.245   7.912   0.798    0.468        1.474
fixed and r50/t5                    0.044  0.121   0.302 -0.271  12.299   0.355    0.406        1.304
fixed or r50/t5                     0.092  0.187   0.491 -0.421  10.836   0.846    0.794        0.964
cells r50/t5 (annual sizing)        0.044  0.145   0.280 -0.364  12.698   0.349    0.531        1.106
joint2f K=2 λ=5 (annual sizing)     0.081  0.192   0.435 -0.379  12.277   0.664    0.847        0.606
joint3f K=4 λ=5 (annual sizing)     0.065  0.119   0.478 -0.205   5.409   1.205    0.391        3.259
selected annually (all candidates)  0.039  0.130   0.257 -0.352  14.720   0.265    0.409        1.779
```

Annual picks: 2009: joint3f K=4 λ=20, 2010: joint2f K=4 λ=20, 2011: joint2f K=4 λ=20, 2012: joint2f K=4 λ=20, 2013: joint2f K=4 λ=20, 2014: joint2f K=4 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=20, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=20, 2022: joint2f K=4 λ=20, 2023: joint3f K=4 λ=5, 2024: joint3f K=4 λ=5, 2025: joint3f K=4 λ=5, 2026: joint3f K=4 λ=5


## EFA

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
EFA                                                                                                  
Buy & hold                          0.072  0.183   0.394 -0.342   9.724   0.740    1.000        0.000
fixed risk λ=50                     0.013  0.154   0.068 -0.505  26.634   0.048    0.711        0.962
fixed trend λ=5                     0.052  0.139   0.335 -0.259   9.937   0.528    0.666        1.082
fixed graded r50/t5                 0.034  0.135   0.212 -0.353  17.651   0.194    0.689        1.022
fixed gate r50/t5                   0.046  0.127   0.305 -0.297  10.276   0.446    0.594        1.022
fixed and r50/t5                    0.038  0.122   0.251 -0.359  12.379   0.307    0.521        0.962
fixed or r50/t5                     0.027  0.168   0.158 -0.517  27.282   0.098    0.856        1.082
cells r50/t5 (annual sizing)        0.015  0.118   0.064 -0.419  20.032   0.074    0.490        1.300
joint2f K=2 λ=5 (annual sizing)     0.034  0.128   0.218 -0.298  15.003   0.230    0.642        0.949
joint3f K=4 λ=5 (annual sizing)     0.042  0.113   0.296 -0.256  10.771   0.391    0.367        3.690
selected annually (all candidates)  0.027  0.109   0.167 -0.306  14.049   0.190    0.317        3.619
```

Annual picks: 2010: joint3f K=4 λ=5, 2011: joint3f K=4 λ=5, 2012: joint3f K=4 λ=5, 2013: joint3f K=4 λ=5, 2014: joint3f K=4 λ=5, 2015: joint3f K=4 λ=5, 2016: joint3f K=4 λ=5, 2017: joint3f K=4 λ=5, 2018: joint2f K=4 λ=5, 2019: joint2f K=4 λ=5, 2020: joint2f K=4 λ=5, 2021: joint3f K=4 λ=5, 2022: joint3f K=4 λ=5, 2023: joint2f K=4 λ=5, 2024: joint2f K=4 λ=5, 2025: joint2f K=4 λ=5, 2026: joint2f K=4 λ=5


## EEM

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
EEM                                                                                                  
Buy & hold                          0.062  0.204   0.319 -0.398  17.028   0.366    1.000        0.000
fixed risk λ=50                     0.027  0.170   0.147 -0.424  18.219   0.149    0.781        0.615
fixed trend λ=5                     0.049  0.129   0.313 -0.290  14.152   0.348    0.504        2.118
fixed graded r50/t5                 0.041  0.130   0.251 -0.281  13.817   0.298    0.642        1.367
fixed gate r50/t5                   0.043  0.112   0.286 -0.219  12.445   0.343    0.441        2.016
fixed and r50/t5                    0.035  0.106   0.224 -0.219  11.887   0.293    0.379        1.913
fixed or r50/t5                     0.041  0.185   0.225 -0.464  21.953   0.189    0.906        0.820
cells r50/t5 (annual sizing)        0.033  0.100   0.212 -0.199   5.014   0.654    0.281        1.084
joint2f K=2 λ=5 (annual sizing)     0.042  0.165   0.234 -0.384  13.158   0.319    0.690        0.614
joint3f K=4 λ=5 (annual sizing)     0.028  0.132   0.156 -0.364  14.191   0.199    0.372        3.043
selected annually (all candidates)  0.052  0.106   0.378 -0.180   3.367   1.542    0.210        1.062
```

Annual picks: 2012: joint2f K=4 λ=20, 2013: joint2f K=4 λ=20, 2014: cells r10/t2, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: cells r10/t20, 2019: cells r10/t20, 2020: cells r10/t20, 2021: cells r100/t2, 2022: cells r100/t2, 2023: cells r100/t2, 2024: cells r100/t2, 2025: cells r100/t2, 2026: cells r100/t2


## TLT

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
TLT                                                                                                  
Buy & hold                          0.020  0.148   0.103 -0.484  22.546   0.087    1.000        0.000
fixed risk λ=50                     0.014  0.096   0.032 -0.211  10.129   0.135    0.614        0.704
fixed trend λ=5                     0.029  0.101   0.187 -0.207  11.852   0.247    0.407        1.407
fixed graded r50/t5                 0.023  0.081   0.135 -0.160   9.029   0.255    0.511        1.055
fixed gate r50/t5                   0.013  0.072   0.010 -0.207  10.834   0.123    0.320        1.439
fixed and r50/t5                   -0.004  0.059  -0.294 -0.207  12.785  -0.032    0.234        1.471
fixed or r50/t5                     0.047  0.126   0.312 -0.213  10.368   0.459    0.787        0.640
cells r50/t5 (annual sizing)        0.007  0.074  -0.068 -0.346  13.355   0.055    0.292        1.228
joint2f K=2 λ=5 (annual sizing)     0.008  0.131   0.010 -0.468  21.328   0.037    0.757        0.576
joint3f K=4 λ=5 (annual sizing)     0.014  0.077   0.029 -0.246  11.067   0.131    0.328        2.275
selected annually (all candidates)  0.013  0.061   0.000 -0.272  10.046   0.133    0.210        0.887
```

Annual picks: 2011: joint3f K=4 λ=20, 2012: cells r10/t2, 2013: cells r50/t2, 2014: cells r50/t2, 2015: cells r50/t2, 2016: cells r20/t2, 2017: cells r20/t2, 2018: cells r20/t2, 2019: cells r20/t2, 2020: cells r100/t20, 2021: cells r100/t20, 2022: cells r50/t20, 2023: cells r50/t20, 2024: cells r50/t20, 2025: cells r50/t20, 2026: cells r50/t20


## GLD

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
GLD                                                                                                  
Buy & hold                          0.069  0.166   0.382 -0.386  19.789   0.349    1.000        0.000
fixed risk λ=50                     0.035  0.114   0.210 -0.385  19.835   0.178    0.582        0.953
fixed trend λ=5                     0.056  0.127   0.358 -0.245  12.559   0.447    0.522        2.126
fixed graded r50/t5                 0.048  0.105   0.332 -0.273  13.442   0.354    0.552        1.539
fixed gate r50/t5                   0.046  0.098   0.329 -0.221   9.610   0.475    0.428        2.016
fixed and r50/t5                    0.033  0.086   0.219 -0.221   9.801   0.336    0.334        1.906
fixed or r50/t5                     0.058  0.148   0.343 -0.385  20.643   0.284    0.771        1.173
cells r50/t5 (annual sizing)        0.008  0.116  -0.022 -0.492  31.683   0.025    0.448        2.055
joint2f K=2 λ=5 (annual sizing)     0.014  0.117   0.031 -0.400  27.647   0.051    0.504        1.005
joint3f K=4 λ=5 (annual sizing)     0.031  0.116   0.173 -0.283  17.360   0.179    0.380        4.489
selected annually (all candidates)  0.035  0.087   0.238 -0.198  11.061   0.315    0.323        1.555
```

Annual picks: 2013: cells r20/t5, 2014: joint2f K=4 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: joint2f K=4 λ=20, 2020: joint2f K=4 λ=20, 2021: joint2f K=4 λ=5, 2022: joint2f K=4 λ=5, 2023: joint2f K=4 λ=5, 2024: joint2f K=4 λ=5, 2025: joint2f K=4 λ=5, 2026: joint2f K=4 λ=5


## XLK

```
                                     CAGR    Vol  Sharpe  MaxDD  Ulcer%  Martin  Time in  Turnover/yr
XLK                                                                                                  
Buy & hold                          0.165  0.234   0.707 -0.531  12.989   1.276    1.000        0.000
fixed risk λ=50                     0.097  0.140   0.625 -0.257   6.205   1.568    0.558        0.560
fixed trend λ=5                     0.128  0.162   0.735 -0.292   7.584   1.696    0.697        2.037
fixed graded r50/t5                 0.115  0.138   0.748 -0.198   5.893   1.951    0.628        1.299
fixed gate r50/t5                   0.108  0.134   0.723 -0.221   6.440   1.687    0.587        1.604
fixed and r50/t5                    0.086  0.123   0.607 -0.221   7.105   1.211    0.477        1.171
fixed or r50/t5                     0.140  0.174   0.752 -0.292   6.933   2.023    0.779        1.426
cells r50/t5 (annual sizing)        0.085  0.138   0.552 -0.229   7.613   1.121    0.546        1.494
joint2f K=2 λ=5 (annual sizing)     0.137  0.173   0.744 -0.273   8.321   1.652    0.780        1.170
joint3f K=4 λ=5 (annual sizing)     0.079  0.104   0.638 -0.234   5.685   1.393    0.354        3.395
selected annually (all candidates)  0.059  0.119   0.411 -0.236   8.098   0.727    0.380        2.546
```

Annual picks: 2007: risk λ=10, 2008: joint3f K=4 λ=5, 2009: joint2f K=4 λ=20, 2010: joint2f K=4 λ=20, 2011: joint2f K=4 λ=20, 2012: joint3f K=4 λ=5, 2013: joint2f K=4 λ=20, 2014: joint2f K=4 λ=20, 2015: joint2f K=4 λ=20, 2016: joint2f K=4 λ=20, 2017: joint2f K=4 λ=20, 2018: joint2f K=4 λ=20, 2019: cells r50/t10, 2020: joint3f K=4 λ=5, 2021: joint3f K=4 λ=5, 2022: joint3f K=4 λ=5, 2023: cells r50/t2, 2024: cells r50/t2, 2025: cells r50/t2, 2026: joint3f K=4 λ=5
