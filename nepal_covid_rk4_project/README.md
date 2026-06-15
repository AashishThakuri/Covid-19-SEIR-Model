# Nepal COVID-19 SEIR RK4 Simple Version

This is the simple version of the assignment code.

It does everything from the screenshot:

- March 2020 to March 2021 simulation.
- Nepal-only COVID-19 data.
- SEIR model.
- `R = recovered + deaths`.
- RK4 method.
- Matplotlib plots using `plt.show()`.

## Main Code

- `nepal_covid_rk4.py`

The code uses simple names:

- `susceptible` = `S`
- `exposed` = `E`
- `infected` = `I`
- `removed` = `R`

Important:

```text
confirmed is NOT susceptible
confirmed = total reported cases
susceptible = population - confirmed
R = recovered + deaths
```

## Outputs

- `data/nepal_covid_mar2020_mar2021.csv`
- `outputs/nepal_daily_seir_parameters.csv`
- `outputs/nepal_seir_rk4_simulation.csv`
- `outputs/summary.json`
- `plots/simple_seir_actual.png`
- `plots/simple_seir_rk4_compare.png`
- `plots/simple_removed_is_recovered_plus_deaths.png`

## Run

```powershell
python .\nepal_covid_rk4.py
```

Or with the bundled Python used in this workspace:

```powershell
& 'C:\Users\LEGION\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\nepal_covid_rk4.py
```
