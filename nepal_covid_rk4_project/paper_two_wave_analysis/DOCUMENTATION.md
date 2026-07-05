# Nepal COVID-19 SEIR RK4 Documentation

## 1. What Was Requested

The assignment image asks for:

1. Use initial values and predict succeeding values with RK4 for both waves.
2. Use different beta values like the reference paper.
3. Add model fitting and validation.
4. Estimate the basic and effective reproduction numbers.

The program completes all four requirements in
`nepal_two_wave_paper_rk4.py` and does not use a notebook.

## 2. Data

The code reads these Johns Hopkins files:

- `time_series_covid19_confirmed_global.csv`
- `time_series_covid19_recovered_global.csv`
- `time_series_covid19_deaths_global.csv`

It filters `Country/Region == "Nepal"`.

Observed compartments are:

```text
S = Nepal population - confirmed
I = confirmed - recovered - deaths
R = recovered + deaths
```

`E` is unavailable in JHU data. It is estimated only for understanding and for
the second-wave initial value:

```text
E = average new confirmed cases during 5 days / eta
```

Since `eta = 0.192`, the average exposed period is:

```text
1 / eta = 5.21 days
```

## 3. Parameters From the Paper

| Parameter | Meaning | Value |
|---|---|---:|
| beta | Transmission rate | 0.0536/day |
| eta | Exposed-to-infectious rate | 0.192/day |
| gamma | Recovery rate | 0.0588/day |
| delta | Disease mortality rate | 0.004/day |
| p | Recorded portion | 0.0473 |

The paper includes natural mortality `mu` in its equations but does not give a
numerical value. The program uses `mu = 0` for the short wave periods instead
of inventing a value.

## 4. Initial Values

### First Wave

The first-wave values are exactly from Table 1 of the paper:

```text
S(0) = 25,000,000
E(0) =    100,000
I(0) =  1,500,000
R(0) =  1,000,000
```

### Second Wave

The paper does not provide second-wave initial values. They are estimated from
JHU data on 14 March 2021. Reported values are divided by `p = 0.0473` to
include estimated unreported infections:

```text
E(0) = reported estimated E / p
I(0) = reported active I / p
R(0) = reported removed R / p
S(0) = Nepal population - E(0) - I(0) - R(0)
```

## 5. Model Equations

The paper model is implemented as:

```text
dS/dt = -beta*S*I/N
dE/dt =  beta*S*I/N - eta*E
dI/dt =  eta*E - (gamma + delta)*I
dR/dt =  (gamma + delta)*I
dL/dt =  p*eta*E
```

`L` is cumulative recorded confirmed cases.

The paper sends recovery into `R` and deaths out of the living population.
This assignment specifically asks for recovered and deaths together in `R`.
Therefore, `(gamma + delta)*I` enters `R`. This leaves the `S`, `E`, and `I`
equations and reproduction-number formula unchanged.

## 6. RK4 Prediction

RK4 calculates four slopes:

```text
k1 = slope at the start
k2 = slope at the first midpoint
k3 = slope at the second midpoint
k4 = slope at the end
```

The next state is:

```text
next = current + (k1 + 2*k2 + 2*k3 + k4) / 6
```

The important correction from the old program is that the model starts once.
Every succeeding value uses the previous model value. It does not restart from
the previous actual value.

## 7. Paper Beta Scenarios

The paper uses these control scenarios:

| Scenario | Beta |
|---|---:|
| No reduction | 0.0536 |
| 25% reduction | 0.0402 |
| 50% reduction | 0.0268 |
| 75% reduction | 0.0134 |

The program runs all four values for both waves. Lower beta produces fewer
predicted cases.

## 8. Model Fitting

A single beta cannot represent both rapid growth and later control. The code
therefore fits:

```text
beta_before_peak
beta_after_peak
```

The change date is the date with the highest daily confirmed cases in each
wave.

The first 80% of the wave is used for fitting. SciPy least squares chooses beta
values that minimize:

```text
sum((model confirmed - actual confirmed)^2)
```

## 9. Validation

The final 20% of each wave is not used to fit beta. It checks prediction on
later dates.

Metrics:

- `RMSE`: typical prediction error, with larger errors penalized more.
- `MAE`: average absolute number of cases missed.
- `MAPE`: average percentage error.
- `R_squared`: how much variation the model explains.

The validation MAPE is approximately:

```text
Wave 1: 4.68%
Wave 2: 2.14%
```

Wave 1 has a negative validation `R_squared` because actual cumulative cases
are nearly flat in the short validation period while the model keeps rising.
This is not a code error. It shows that the simple two-beta SEIR model does not
capture every reporting and control change.

## 10. Reproduction Numbers

The paper derives:

```text
R0 = eta*beta / ((eta + mu)*(mu + delta + gamma))
```

Because `mu = 0`, this simplifies to:

```text
R0 = beta / (gamma + delta)
```

The effective reproduction number is:

```text
Re(t) = R0(t) * S(t) / N(t)
```

Meaning:

- `R0 > 1`: infection can grow when almost everyone is susceptible.
- `R0 < 1`: each infectious person produces less than one new case on average.
- `Re > 1`: the epidemic tends to grow at that time.
- `Re < 1`: the epidemic tends to shrink at that time.

## 11. Main Findings

1. The second wave is much larger than the first wave.
2. Fitted beta before the peak is about `0.0632` in wave 1.
3. Fitted beta before the peak is about `0.2631` in wave 2.
4. The paper beta `0.0536` gives `R0 = 0.8535`.
5. The paper beta under-predicts the second wave, so the same transmission rate
   cannot describe both waves.
6. Beta becomes smaller after both peaks.
7. All paper beta reductions produce fewer predicted cases.
8. `Re` falls sharply when beta changes after the peak.

## 12. Output Interpretation

`summary.json` contains the actual findings, fitted beta values, validation
metrics, reproduction numbers, initial values, scenario end values, and a
simple conclusion for every plot.

The CSV files keep all daily values so each numerical result can be checked.
