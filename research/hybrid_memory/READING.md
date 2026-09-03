# Reading (written after the run; rule frozen before it)

**Both RETIRE.** The trend hybrid lands between the two arms it was built from on every measure: Martin 0.57 (rolling 0.59, expanding 0.49), MaxDD −32% (−29%, −32%), 2022 fold −7.1% (−3.2%, −10.1%), 2023–24 off-fraction 0.33 (0.39, 0.28). Sharpe is a wash (+0.02, 5 of 8), Martin −0.09 (3 of 8), bleed passes. Exactly the "between the two failure modes" outcome the calibration reading predicted for an average of memories, now measured for the state-dependent version too.

**The risk hybrid did not reproduce the expanding cure.** SPY time in market 2019–2022 is 0.04 for the hybrid against 0.41 for expanding and 0.00 for rolling, even though the hybrid uses the expanding scaler and calm centre. So the cure in the expanding arm came from the *diluted crisis centre*, not from a better calm centre — which means anchoring and slowness were never two effects to be separated; they are one effect seen from two sides. Where the crisis centre sits decides both how far the calm state extends and how fast the model leaves it.

**Disposition.** `calibration="rolling"` stays the default; the memory question is closed on this evidence for both models. A future design would need to change what is compared (for example, a threshold set in volatility units rather than fitted centres), not how long the memory is.
