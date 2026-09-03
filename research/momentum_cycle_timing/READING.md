# Reading (written after the run; rule frozen before it)

**All five arms RETIRE against the trend switch, each losing Martin in 7 of 8 funds with the CI excluding zero.** Full-sample medians: trend λ5 Sharpe 0.38 / MaxDD −29% / Martin 0.59 / 1.7 flips a year; the best cycle arm (OutOnlyBear) 0.40 / −46% / 0.46 / 9.7 flips; the paper's dynamic-speed rule 0.35 / −39% / 0.39 / 12.2 flips. None of the cycle arms beats buy-and-hold on Sharpe in more than one fund.

**Why.** The four-state labels flip with the raw signs of two trailing means, and at daily frequency those signs flip constantly: 24 to 29 flips a year for the fast-sign arms, 5.6 even for the slow sign alone. The jump model reads the same two signals but charges a toll to change state, so it flips 1.7 times a year and holds its bear exits (2022 MaxDD −3% against −16% to −23% for every cycle arm). The cycle arms' extra drawdown is not a cost of being in the wrong state; it is the cost of re-entering on every fast-sign bounce during a bear.

**The dynamic speeds collapsed to a fixed rule.** Estimated causally each year, a_Co was 0 in every year (stay long through Corrections) and a_Re rose from 0.5 to 1.0 from 2018 onward (go fully long in Rebounds). From 2018 the DYN arm is therefore identical to OutOnlyBear, which is why their fold statistics match. The paper's Sharpe-maximising speeds, applied long-only at daily frequency, say "ignore the fast signal", which is the SlowSign arm with a different Rebound treatment.

**The information content did not survive the horizon change.** The paper's separation of state means is a monthly result on the market factor. At daily frequency and fund level the next-day means by state are noisy and unordered (SPY: Bull 8.9%, Correction 20.7%, Bear 13.1%, Rebound 7.8% annualised), so there is nothing for a daily rule to harvest. The regimes module keeps its value as a monthly descriptive and conditional-averages tool; as a daily switch it is dominated by the jump model on the same inputs.

**Disposition.** Trend λ5 rolling remains the switch; the momentum-cycle question is closed for daily timing.
