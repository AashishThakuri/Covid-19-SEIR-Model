# Nepal COVID-19 SEIR RK4 - Both Waves

This project follows the supplied reference paper, **Transmission Dynamics of
COVID-19 in Nepal** by Adhikari et al. (2021), and completes every item in the
assignment image.

## Completed Tasks

- Uses one initial state and predicts succeeding values with RK4.
- Simulates Nepal's first and second COVID-19 waves.
- Uses the paper beta `0.0536`.
- Tests beta reductions of 25%, 50%, and 75%.
- Fits beta by least squares.
- Uses the final 20% of each wave for model validation.
- Calculates the basic reproduction number `R0`.
- Calculates the effective reproduction number `Re`.
- Uses `R = recovered + deaths`.
- Uses Matplotlib and `plt.show()`.
- Does not create a notebook.

## Wave Periods

| Wave | Dates | Reason |
|---|---|---|
| First wave | 17 Sep 2020 to 13 Mar 2021 | Exact study period in the paper |
| Second wave | 14 Mar 2021 to 31 Jul 2021 | Covers Nepal's second-wave peak while JHU recovery data is still available |

## Run

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Run the program:

```powershell
python .\nepal_two_wave_paper_rk4.py
```

The program saves every plot and then displays it with `plt.show()`.

## Important Outputs

- `data/nepal_covid_two_waves.csv`
- `outputs/paper_beta_scenarios_both_waves.csv`
- `outputs/rk4_model_fitting_validation.csv`
- `outputs/model_validation_metrics.csv`
- `outputs/model_parameters.csv`
- `outputs/summary.json`
- `plots/both_waves_actual.png`
- `plots/removed_is_recovered_plus_deaths.png`
- `plots/wave_1_paper_beta_scenarios.png`
- `plots/wave_2_paper_beta_scenarios.png`
- `plots/model_fitting_and_validation.png`
- `plots/basic_and_effective_reproduction_numbers.png`

## Main Result

The fitted transmission rate before the peak is about `0.0632` in wave 1 and
`0.2631` in wave 2. Therefore, the paper's beta `0.0536` cannot explain the
rapid growth of the second wave. In both waves, fitted beta becomes smaller
after the peak.
