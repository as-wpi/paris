# Cross-asset walk-forward summary — 2026-09-02

Out of sample from the first selection year per series (MKT 1995, SPY 2002, QQQ 2008, XLK 2007, IWM 2009, EFA 2010, EEM 2012, TLT 2011, GLD 2013) to 2026. Execution: t+0 (state known at close T-1, applied to day T), 10 bp one-way on exposure changes, cash = 1-month T-bill. Rows marked (causal ...) re-estimate sizing or re-select the rule each year on history to the prior year-end only. The default penalties (risk 50, trend 5) were sized in sample on MKT, so MKT's fixed rows are not clean evidence; the eight ETFs are untouched by any tuning.


## Sharpe

```
                                         MKT    SPY    QQQ    XLK    IWM    EFA    EEM    TLT    GLD  mean (ETFs)  wins vs B&H (of 8)
B&H                                    0.537  0.509  0.716  0.707  0.555  0.394  0.319  0.103  0.382        0.461                   0
risk λ50                               0.595  0.467  0.669  0.625  0.309  0.068  0.145  0.063  0.238        0.323                   0
trend λ5                               0.762  0.723  0.902  0.714  0.501  0.345  0.328  0.191  0.362        0.508                   5
graded                                 0.778  0.671  0.859  0.736  0.446  0.216  0.256  0.155  0.351        0.461                   4
gate                                   0.778  0.713  0.812  0.699  0.427  0.313  0.305  0.037  0.369        0.459                   2
and                                    0.645  0.591  0.602  0.580  0.295  0.257  0.246 -0.233  0.300        0.330                   1
or                                     0.738  0.628  0.945  0.752  0.481  0.161  0.221  0.312  0.322        0.478                   4
cells (causal sizing)                  0.624  0.652  0.759  0.557  0.263  0.107  0.224 -0.170  0.035        0.303                   2
joint (logvol,slow) K=2 (causal)       0.622  0.664  0.863  0.748  0.439  0.218  0.272  0.020  0.074        0.412                   3
joint (logvol,slow,fast) K=4 (causal)  0.533  0.348  0.474  0.740  0.519  0.172  0.308 -0.024  0.313        0.356                   1
selected annually (causal)             0.615  0.488  0.598  0.324  0.389  0.188  0.327  0.368  0.239        0.365                   2
```


## MaxDD

```
                                         MKT    SPY    QQQ    XLK    IWM    EFA    EEM    TLT    GLD  mean (ETFs)  wins vs B&H (of 8)
B&H                                   -0.546 -0.552 -0.500 -0.531 -0.411 -0.342 -0.398 -0.484 -0.386       -0.450                   0
risk λ50                              -0.219 -0.317 -0.228 -0.257 -0.494 -0.505 -0.427 -0.211 -0.385       -0.353                   5
trend λ5                              -0.203 -0.186 -0.195 -0.292 -0.289 -0.259 -0.298 -0.207 -0.245       -0.246                   8
graded                                -0.187 -0.186 -0.176 -0.198 -0.297 -0.350 -0.281 -0.160 -0.253       -0.238                   7
gate                                  -0.187 -0.186 -0.194 -0.221 -0.245 -0.312 -0.208 -0.207 -0.179       -0.219                   8
and                                   -0.198 -0.186 -0.194 -0.221 -0.274 -0.372 -0.208 -0.207 -0.179       -0.230                   7
or                                    -0.247 -0.317 -0.228 -0.292 -0.421 -0.512 -0.470 -0.213 -0.385       -0.355                   5
cells (causal sizing)                 -0.245 -0.186 -0.194 -0.211 -0.364 -0.386 -0.189 -0.364 -0.358       -0.282                   7
joint (logvol,slow) K=2 (causal)      -0.318 -0.186 -0.208 -0.268 -0.401 -0.298 -0.344 -0.472 -0.354       -0.316                   8
joint (logvol,slow,fast) K=4 (causal) -0.299 -0.259 -0.293 -0.184 -0.213 -0.355 -0.303 -0.198 -0.256       -0.258                   7
selected annually (causal)            -0.187 -0.256 -0.228 -0.270 -0.296 -0.282 -0.201 -0.145 -0.196       -0.234                   8
```


## Martin

```
                                         MKT    SPY    QQQ    XLK    IWM    EFA    EEM    TLT    GLD  mean (ETFs)  wins vs B&H (of 8)
B&H                                    0.748  0.794  1.340  1.276  1.054  0.740  0.366  0.087  0.349        0.751                   0
risk λ50                               1.474  0.885  1.787  1.568  0.223  0.048  0.144  0.164  0.194        0.627                   4
trend λ5                               2.303  1.964  2.628  1.646  1.079  0.555  0.386  0.252  0.478        1.124                   7
graded                                 2.609  1.931  2.723  1.919  0.714  0.199  0.311  0.274  0.390        1.058                   5
gate                                   2.869  2.320  2.272  1.629  0.766  0.466  0.397  0.141  0.615        1.076                   6
and                                    2.018  1.945  1.281  1.160  0.334  0.323  0.350 -0.005  0.491        0.735                   2
or                                     1.835  1.206  3.145  2.023  0.824  0.101  0.181  0.459  0.265        1.026                   4
cells (causal sizing)                  1.258  1.980  2.410  1.152  0.413  0.114  0.732 -0.008  0.068        0.858                   3
joint (logvol,slow) K=2 (causal)       0.990  1.644  1.879  1.650  0.703  0.230  0.412  0.044  0.086        0.831                   4
joint (logvol,slow,fast) K=4 (causal)  0.985  0.480  0.874  1.602  1.124  0.153  0.395  0.108  0.330        0.633                   4
selected annually (causal)             1.902  0.845  1.367  0.516  0.692  0.285  1.024  0.679  0.338        0.718                   4
```


## CAGR

```
                                         MKT    SPY    QQQ    XLK    IWM    EFA    EEM    TLT    GLD  mean (ETFs)  wins vs B&H (of 8)
B&H                                    0.114  0.100  0.160  0.165  0.121  0.072  0.062  0.020  0.069        0.096                   0
risk λ50                               0.081  0.063  0.098  0.097  0.052  0.013  0.027  0.017  0.038        0.051                   0
trend λ5                               0.113  0.099  0.145  0.125  0.079  0.054  0.052  0.030  0.057        0.080                   1
graded                                 0.098  0.082  0.123  0.113  0.068  0.035  0.042  0.025  0.049        0.067                   1
gate                                   0.095  0.083  0.113  0.105  0.062  0.047  0.045  0.015  0.050        0.065                   0
and                                    0.075  0.066  0.080  0.082  0.043  0.039  0.037 -0.001  0.040        0.048                   0
or                                     0.119  0.095  0.164  0.140  0.089  0.027  0.041  0.047  0.055        0.082                   2
cells (causal sizing)                  0.093  0.074  0.094  0.081  0.038  0.020  0.034 -0.001  0.015        0.044                   0
joint (logvol,slow) K=2 (causal)       0.105  0.089  0.143  0.137  0.084  0.034  0.047  0.009  0.019        0.070                   0
joint (logvol,slow,fast) K=4 (causal)  0.082  0.049  0.079  0.102  0.073  0.028  0.046  0.012  0.046        0.054                   0
selected annually (causal)             0.086  0.065  0.084  0.049  0.054  0.029  0.048  0.042  0.034        0.051                   1
```


## Turnover/yr

```
                                        MKT   SPY   QQQ   XLK   IWM   EFA   EEM   TLT   GLD  mean (ETFs)  wins vs B&H (of 8)
B&H                                    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00         0.00                   0
risk λ50                               0.76  0.61  0.70  0.56  0.85  0.96  0.75  0.70  0.95         0.76                   8
trend λ5                               1.81  1.34  1.77  2.04  1.42  0.96  2.53  1.54  1.98         1.70                   8
graded                                 1.29  0.97  1.23  1.30  1.08  0.96  1.64  1.12  1.47         1.22                   8
gate                                   1.27  1.10  1.66  1.60  1.36  0.96  2.49  1.57  1.94         1.59                   8
and                                    0.73  0.85  1.56  1.17  1.30  0.96  2.46  1.60  1.91         1.48                   8
or                                     1.84  1.10  0.91  1.43  0.85  0.96  0.82  0.64  1.03         0.97                   8
cells (causal sizing)                  1.27  1.01  1.25  1.44  0.95  1.27  1.22  1.46  2.35         1.37                   8
joint (logvol,slow) K=2 (causal)       0.78  0.93  1.11  1.30  0.64  0.95  0.70  0.63  1.16         0.93                   8
joint (logvol,slow,fast) K=4 (causal)  3.89  2.60  3.37  4.23  3.32  3.68  3.34  2.30  4.17         3.38                   8
selected annually (causal)             1.44  1.05  2.01  2.20  1.82  2.95  1.27  1.67  1.60         1.82                   8
```


## Reading

- Trend alone (λ=5) is the best single rule: Sharpe above buy-and-hold on 5 of 8 ETFs (mean 0.51 vs 0.46), max drawdown better on 8 of 8, Martin on 7 of 8.
- The combinations buy drawdown protection, not Sharpe: gate has the best mean max drawdown (−22 % vs −45 %) at buy-and-hold Sharpe; graded is similar. AND is too restrictive; OR gives the protection back.
- Risk alone is weak outside US equities (EFA, EEM, TLT, GLD): a volatility gate on assets whose high-vol periods are not systematically down.
- Annual re-selection over ~100 candidates is harmful (Sharpe above buy-and-hold on 2 of 8): selection noise exceeds the differences between rules. Fix the rule ex ante.
- The joint multi-feature models with causal sizing do not beat the two binaries; the four-state three-feature model turns over ~3.4 units/yr for nothing.
- Every rule gives up CAGR (mean 6.5–8.0 % vs 9.6 %): these are drawdown gates. Sized at equal risk they would compare on Sharpe/Martin, which is where trend and gate lead.
