# Simple Code Explanation

This guide explains what each important line or small group of lines in
`nepal_two_wave_paper_rk4.py` does, why it is needed, and what would happen
without it.

## Imports

| Code | What it does | Why it is needed |
|---|---|---|
| `import json` | Works with JSON text. | Saves the result-focused `summary.json`. |
| `from pathlib import Path` | Handles file and folder paths. | Makes paths work cleanly after moving the project. |
| `import matplotlib.pyplot as plt` | Creates and displays graphs. | Required for saved plots and `plt.show()`. |
| `import numpy as np` | Works with numerical arrays. | RK4 state values and calculations use arrays. |
| `import pandas as pd` | Reads and organizes CSV data. | JHU data and output tables are DataFrames. |
| `from scipy.optimize import least_squares` | Finds beta values with the smallest squared error. | Required for model fitting. |

Without an import, the names supplied by that package would not exist and the
program would stop with a `NameError`.

## Project Folders

```python
PROJECT_FOLDER = Path(__file__).resolve().parent
```

`__file__` is the Python file. `resolve()` gets its complete location and
`parent` gets the containing folder. This lets the project work after it is
moved.

```python
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
PLOT_FOLDER = PROJECT_FOLDER / "plots"
```

These are the output locations. Without them, every save statement would need
a repeated hard-coded path.

## Finding the JHU Data

```python
search_folders = [PROJECT_FOLDER] + list(PROJECT_FOLDER.parents)
```

This creates a list containing the project folder and each parent folder.

```python
for folder in search_folders:
```

This checks the possible locations one by one.

```python
if confirmed_file.exists():
    return possible_folder
```

When the confirmed CSV exists, that location is returned. Without this search,
moving the semester project could break the script.

```python
raise FileNotFoundError(...)
```

This gives a clear error when the cloned repository is missing. Without it,
the user would receive a less helpful error later.

## Dates and Waves

```python
DATA_START = "2020-03-01"
DATA_END = "2021-07-31"
```

These limit the data. July 31 is used because the JHU recovered series stops
being reliable in early August 2021.

```python
WAVES = {...}
```

This stores both wave names and date ranges in one place. Without it, the same
dates would be repeated in many functions.

## Paper Parameters

```python
PAPER_BETA = 0.0536
ETA = 0.192
GAMMA = 0.0588
DELTA = 0.004
RECORDED_PORTION = 0.0473
```

These values come from Table 1 of the supplied paper.

```python
MU = 0.0
```

The paper uses `mu` but gives no numerical value. Zero is used for these short
periods. Without defining `MU`, the paper's reproduction formula could not be
written directly in code.

```python
BETA_SCENARIOS = {...}
```

This stores the original beta and the paper's 25%, 50%, and 75% reductions.

## Reading One JHU File

```python
all_data = pd.read_csv(COVID_DATA_FOLDER / file_name)
```

Reads the CSV into a table.

```python
nepal_rows = all_data[all_data["Country/Region"] == "Nepal"]
```

Keeps Nepal only. Without this filter, values from every country would be used.

```python
date_columns = all_data.columns[4:]
```

The first four columns describe locations. All later columns are dates.

```python
.apply(pd.to_numeric, errors="coerce")
```

Converts values to numbers. Invalid text becomes missing data instead of
crashing.

```python
.fillna(0)
```

Replaces missing numbers with zero.

```python
.sum(axis=0)
```

Adds all Nepal rows for each date. This is safe even if a country has multiple
province rows.

```python
pd.to_datetime(date_columns, format="%m/%d/%y")
```

Converts column labels such as `3/14/21` into real dates.

## Making S, E, I, and R

```python
data["removed_R"] = data["recovered"] + data["deaths"]
```

Implements the assignment rule `R = recovered + deaths`.

```python
data["infected_I"] = data["confirmed"] - data["removed_R"]
```

Calculates active infected cases.

```python
data["susceptible_S"] = NEPAL_POPULATION - data["confirmed"]
```

Estimates people not yet counted as confirmed.

```python
data["new_confirmed"] = data["confirmed"].diff()
```

Subtracts yesterday's cumulative cases from today's cumulative cases.

```python
average_new_cases = data["new_confirmed"].rolling(window=5).mean()
```

Smooths reporting noise by averaging five recent days.

```python
data["estimated_exposed_E"] = average_new_cases / ETA
```

Estimates E because JHU does not provide exposed people. Without an E estimate,
the second-wave initial E could not be formed from the data.

## Initial Values

```python
if wave_key == "wave_1":
```

Selects the first-wave rule.

```python
susceptible = 25_000_000.0
exposed = 100_000.0
infected = 1_500_000.0
removed = 1_000_000.0
```

These are the exact initial values in the paper.

For wave 2:

```python
exposed = first["estimated_exposed_E"] / RECORDED_PORTION
infected = first["infected_I"] / RECORDED_PORTION
removed = first["removed_R"] / RECORDED_PORTION
```

Dividing by `p` estimates reported plus unreported infections.

```python
susceptible = NEPAL_POPULATION - exposed - infected - removed
```

Makes the second-wave compartments add to Nepal's population.

```python
return np.array([susceptible, exposed, infected, removed, recorded_cases])
```

Places all model values in one numeric array used by RK4.

## SEIR Changes

```python
new_exposed = beta * susceptible * infected / population
```

Calculates movement from S to E.

```python
new_infected = ETA * exposed
```

Calculates movement from E to I.

```python
new_removed = (GAMMA + DELTA) * infected
```

Calculates movement from I to R, including recovery and death.

```python
dL = RECORDED_PORTION * new_infected
```

This is the paper's recorded cumulative case equation `L'(t) = p*eta*E`.

```python
return np.array([dS, dE, dI, dR, dL])
```

Returns all slopes together. RK4 needs these slopes.

## RK4

```python
k1 = seir_changes(state, beta)
k2 = seir_changes(state + 0.5 * step_size * k1, beta)
k3 = seir_changes(state + 0.5 * step_size * k2, beta)
k4 = seir_changes(state + step_size * k3, beta)
```

These are four slope estimates at the start, two midpoints, and end.

```python
next_state = state + (step_size / 6.0) * (
    k1 + 2 * k2 + 2 * k3 + k4
)
```

Combines the four slopes using the RK4 formula.

```python
return np.maximum(next_state, 0.0)
```

Prevents impossible negative population values.

## Forward Simulation

```python
states[0] = initial_state
```

Sets the starting state once.

```python
for day in range(1, len(dates)):
    states[day] = rk4_step(states[day - 1], beta_values[day - 1])
```

Each day uses the previous predicted day. This is true forward prediction.
Without this loop, there is no simulation.

## Beta Scenarios

```python
for scenario_name, beta in BETA_SCENARIOS.items():
```

Runs every beta requested from the paper.

```python
beta_values = np.full(len(wave_data), beta)
```

Uses the chosen beta on every date in that scenario.

## Fitting

```python
change_day = int(wave_data["new_confirmed"].idxmax())
```

Finds the observed daily-case peak. Beta changes there.

```python
split_day = int(len(wave_data) * 0.80)
```

Uses 80% for fitting and leaves 20% for validation.

```python
return (model_confirmed - actual) / 1000.0
```

Returns residual errors to the optimizer. Scaling by 1000 improves numerical
stability but does not change the best beta.

```python
fitted = least_squares(...)
```

Searches for the two beta values with the smallest squared residuals.

## Validation Metrics

```python
rmse = np.sqrt(np.mean(errors ** 2))
```

Calculates root mean squared error.

```python
mae = np.mean(np.abs(errors))
```

Calculates average absolute error.

```python
mape = np.mean(np.abs(errors / actual)) * 100
```

Calculates average percentage error.

```python
r_squared = 1 - error_variation / actual_variation
```

Measures how much variation the model explains.

## Reproduction Numbers

```python
R0 = ETA * beta / ((ETA + MU) * (MU + DELTA + GAMMA))
```

This is the paper's basic reproduction number formula.

```python
Re = R0 * S / N
```

This adjusts R0 for the susceptible proportion at each date.

Without these calculations, the assignment's reproduction-number requirement
would not be completed.

## Saving Files

```python
file_path.write_text(table.to_csv(index=False), encoding="utf-8")
```

Converts the CSV to text first and then saves it. This avoids the previous
Windows/OneDrive bad file descriptor problem.

```python
json.dumps(summary, indent=2)
```

Turns the findings dictionary into readable JSON.

## Plots

```python
fig.savefig(...)
```

Saves a permanent PNG file.

```python
plt.show()
```

Displays the Matplotlib window when `show_plots=True`.

## Main Function

```python
def main(show_plots=True):
```

Groups the complete workflow in one function.

```python
if __name__ == "__main__":
    main(show_plots=True)
```

Runs the workflow only when this file is executed directly. Without this
condition, importing the file for testing would immediately open every plot.
