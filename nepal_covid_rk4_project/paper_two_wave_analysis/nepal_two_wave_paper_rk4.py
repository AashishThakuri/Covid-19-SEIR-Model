"""Nepal COVID-19 SEIR model for two waves using the RK4 method.

The model and parameter values come from the supplied reference paper:
"Transmission Dynamics of COVID-19 in Nepal" (Adhikari et al., 2021).

This program:
1. Reads Johns Hopkins COVID-19 data and keeps Nepal only.
2. Studies the first and second Nepal waves.
3. Predicts every succeeding value from one initial value using RK4.
4. Tests the paper beta and its 25%, 50%, and 75% reductions.
5. Fits two beta values for each wave with least squares.
6. Validates the fitted model on the final 20% of each wave.
7. Calculates the basic and effective reproduction numbers.
8. Combines recovered and deaths into the SEIR R compartment.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# -------------------------------------------------------------------
# 1. Folders and dates
# -------------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
PLOT_FOLDER = PROJECT_FOLDER / "plots"


def find_covid_data_folder():
    """Find the cloned Johns Hopkins time-series data folder."""

    search_folders = [PROJECT_FOLDER] + list(PROJECT_FOLDER.parents)

    for folder in search_folders:
        possible_folder = (
            folder
            / "COVID-19"
            / "csse_covid_19_data"
            / "csse_covid_19_time_series"
        )
        confirmed_file = possible_folder / "time_series_covid19_confirmed_global.csv"

        if confirmed_file.exists():
            return possible_folder

    raise FileNotFoundError(
        "COVID-19 data was not found. Keep the cloned COVID-19 folder "
        "inside the semester project folder."
    )


COVID_DATA_FOLDER = find_covid_data_folder()

# The second wave ends on 31 July because the JHU recovered series for Nepal
# stops being maintained in early August 2021.
DATA_START = "2020-03-01"
DATA_END = "2021-07-31"

WAVES = {
    "wave_1": {
        "name": "First wave",
        "start": "2020-09-17",
        "end": "2021-03-13",
    },
    "wave_2": {
        "name": "Second wave",
        "start": "2021-03-14",
        "end": "2021-07-31",
    },
}


# -------------------------------------------------------------------
# 2. Parameters taken from the reference paper
# -------------------------------------------------------------------

NEPAL_POPULATION = 29_136_808

PAPER_BETA = 0.0536       # Transmission rate per day
ETA = 0.192               # E to I rate per day
GAMMA = 0.0588            # Recovery rate per day
DELTA = 0.004              # Disease death rate per day
RECORDED_PORTION = 0.0473  # Proportion of infections recorded as cases

# The paper includes natural death (mu) and recruitment in its equations,
# but it does not give their numerical values. For these short wave periods,
# both are set to zero instead of inventing an unsupported number.
MU = 0.0

# Exact beta reductions used by the paper.
BETA_SCENARIOS = {
    "Paper beta (0% reduction)": PAPER_BETA,
    "25% beta reduction": PAPER_BETA * 0.75,
    "50% beta reduction": PAPER_BETA * 0.50,
    "75% beta reduction": PAPER_BETA * 0.25,
}


# -------------------------------------------------------------------
# 3. Read and prepare Nepal data
# -------------------------------------------------------------------

def read_nepal_series(file_name):
    """Read one JHU CSV and return Nepal's cumulative values by date."""

    all_data = pd.read_csv(COVID_DATA_FOLDER / file_name)
    nepal_rows = all_data[all_data["Country/Region"] == "Nepal"]

    if nepal_rows.empty:
        raise ValueError("Nepal was not found in " + file_name)

    date_columns = all_data.columns[4:]
    values = (
        nepal_rows[date_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=0)
    )

    return pd.Series(
        values.to_numpy(dtype=float),
        index=pd.to_datetime(date_columns, format="%m/%d/%y"),
    ).sort_index()


def make_nepal_data():
    """Create the observed S, E, I, and R values used in this project."""

    confirmed = read_nepal_series("time_series_covid19_confirmed_global.csv")
    recovered = read_nepal_series("time_series_covid19_recovered_global.csv")
    deaths = read_nepal_series("time_series_covid19_deaths_global.csv")

    data = pd.DataFrame(
        {
            "confirmed": confirmed,
            "recovered": recovered,
            "deaths": deaths,
        }
    ).loc[DATA_START:DATA_END]

    data = data.ffill().fillna(0)

    # Assignment rule: R contains everyone removed from active infection.
    data["removed_R"] = data["recovered"] + data["deaths"]
    data["infected_I"] = (
        data["confirmed"] - data["removed_R"]
    ).clip(lower=0)
    data["susceptible_S"] = (
        NEPAL_POPULATION - data["confirmed"]
    ).clip(lower=0)
    data["new_confirmed"] = (
        data["confirmed"].diff().fillna(0).clip(lower=0)
    )

    # JHU does not contain exposed E. ETA = 0.192 means that an exposed
    # person takes about 1 / 0.192 = 5.21 days to become infectious.
    average_new_cases = data["new_confirmed"].rolling(
        window=5,
        min_periods=1,
    ).mean()
    data["estimated_exposed_E"] = average_new_cases / ETA

    data.insert(0, "date", data.index)
    data.insert(1, "day", np.arange(len(data)))
    return data.reset_index(drop=True)


def get_wave_data(data, wave_key):
    """Return only the dates belonging to one wave."""

    wave = WAVES[wave_key]
    dates = pd.to_datetime(data["date"])
    selected = (dates >= wave["start"]) & (dates <= wave["end"])
    return data.loc[selected].reset_index(drop=True)


# -------------------------------------------------------------------
# 4. Initial values for both waves
# -------------------------------------------------------------------

def get_initial_state(wave_key, wave_data):
    """Return initial [S, E, I, R, L] for one wave.

    L is the cumulative number of recorded confirmed cases.
    """

    first = wave_data.iloc[0]

    if wave_key == "wave_1":
        # These four values are exactly Table 1 of the reference paper.
        susceptible = 25_000_000.0
        exposed = 100_000.0
        infected = 1_500_000.0
        removed = 1_000_000.0
    else:
        # The paper does not provide second-wave initial values.
        # Divide reported values by p to estimate total reported + unreported.
        exposed = first["estimated_exposed_E"] / RECORDED_PORTION
        infected = first["infected_I"] / RECORDED_PORTION
        removed = first["removed_R"] / RECORDED_PORTION
        susceptible = NEPAL_POPULATION - exposed - infected - removed
        susceptible = max(susceptible, 0.0)

    recorded_cases = float(first["confirmed"])

    return np.array(
        [susceptible, exposed, infected, removed, recorded_cases],
        dtype=float,
    )


# -------------------------------------------------------------------
# 5. Paper SEIR equations and RK4
# -------------------------------------------------------------------

def seir_changes(state, beta):
    """Calculate dS/dt, dE/dt, dI/dt, dR/dt, and dL/dt."""

    susceptible, exposed, infected, removed, recorded_cases = state
    population = max(susceptible + exposed + infected + removed, 1.0)

    new_exposed = beta * susceptible * infected / population
    new_infected = ETA * exposed
    new_removed = (GAMMA + DELTA) * infected
    natural_deaths = MU

    # Recruitment equals natural deaths so natural population stays stable.
    recruitment = natural_deaths * population

    dS = recruitment - new_exposed - natural_deaths * susceptible
    dE = new_exposed - new_infected - natural_deaths * exposed
    dI = new_infected - new_removed - natural_deaths * infected

    # The paper sends gamma*I to recovered and delta*I to deaths.
    # The assignment asks for recovered + deaths in R, so both enter R here.
    dR = new_removed - natural_deaths * removed

    # Paper fitting equation: L'(t) = p * eta * E.
    dL = RECORDED_PORTION * new_infected

    return np.array([dS, dE, dI, dR, dL], dtype=float)


def rk4_step(state, beta, step_size=1.0):
    """Move the model one day forward with fourth-order Runge-Kutta."""

    k1 = seir_changes(state, beta)
    k2 = seir_changes(state + 0.5 * step_size * k1, beta)
    k3 = seir_changes(state + 0.5 * step_size * k2, beta)
    k4 = seir_changes(state + step_size * k3, beta)

    next_state = state + (step_size / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )

    return np.maximum(next_state, 0.0)


def simulate_rk4(initial_state, dates, beta_values):
    """Predict all succeeding dates from one initial state."""

    beta_values = np.asarray(beta_values, dtype=float)
    states = np.zeros((len(dates), 5), dtype=float)
    states[0] = initial_state

    for day in range(1, len(dates)):
        states[day] = rk4_step(states[day - 1], beta_values[day - 1])

    result = pd.DataFrame(
        states,
        columns=[
            "model_S",
            "model_E",
            "model_I",
            "model_R",
            "model_confirmed",
        ],
    )
    result.insert(0, "date", pd.to_datetime(dates).to_numpy())
    result["beta"] = beta_values
    return result


# -------------------------------------------------------------------
# 6. Paper beta scenarios
# -------------------------------------------------------------------

def run_paper_beta_scenarios(data):
    """Run all four paper beta values for both waves."""

    all_scenarios = []

    for wave_key in WAVES:
        wave_data = get_wave_data(data, wave_key)
        initial_state = get_initial_state(wave_key, wave_data)

        for scenario_name, beta in BETA_SCENARIOS.items():
            beta_values = np.full(len(wave_data), beta)
            result = simulate_rk4(
                initial_state,
                wave_data["date"],
                beta_values,
            )
            result.insert(0, "wave", wave_key)
            result.insert(1, "scenario", scenario_name)
            result["actual_confirmed"] = wave_data["confirmed"]
            result["basic_R0"] = basic_reproduction_number(beta)
            result["effective_Re"] = effective_reproduction_number(
                result,
                beta_values,
            )
            all_scenarios.append(result)

    return pd.concat(all_scenarios, ignore_index=True)


# -------------------------------------------------------------------
# 7. Model fitting and validation
# -------------------------------------------------------------------

def beta_schedule(length, change_day, beta_before, beta_after):
    """Use one beta before the wave peak and another beta after it."""

    values = np.full(length, beta_after, dtype=float)
    values[: change_day + 1] = beta_before
    return values


def calculate_metrics(actual, predicted):
    """Calculate simple model fitting or validation measurements."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    nonzero = actual != 0
    mape = float(
        np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100
    )

    total_variation = np.sum((actual - actual.mean()) ** 2)
    if total_variation == 0:
        r_squared = 0.0
    else:
        r_squared = float(
            1 - np.sum(errors ** 2) / total_variation
        )

    return {
        "RMSE_cases": rmse,
        "MAE_cases": mae,
        "MAPE_percent": mape,
        "R_squared": r_squared,
    }


def fit_one_wave(data, wave_key):
    """Fit two beta values and validate on the final 20% of a wave."""

    wave_data = get_wave_data(data, wave_key)
    dates = wave_data["date"]
    actual = wave_data["confirmed"].to_numpy(dtype=float)
    initial_state = get_initial_state(wave_key, wave_data)

    # A wave grows before its largest daily case count and slows afterward.
    change_day = int(wave_data["new_confirmed"].idxmax())
    change_date = pd.Timestamp(wave_data.loc[change_day, "date"])

    split_day = int(len(wave_data) * 0.80)
    split_day = min(max(split_day, change_day + 2), len(wave_data) - 1)

    def residuals(beta_pair):
        beta_values = beta_schedule(
            split_day,
            change_day,
            beta_pair[0],
            beta_pair[1],
        )
        model = simulate_rk4(
            initial_state,
            dates.iloc[:split_day],
            beta_values,
        )

        # Scaling keeps the optimizer's numbers small and stable.
        return (
            model["model_confirmed"].to_numpy() - actual[:split_day]
        ) / 1000.0

    fitted = least_squares(
        residuals,
        x0=[0.08, 0.04],
        bounds=([0.001, 0.001], [0.5, 0.5]),
    )

    beta_before, beta_after = fitted.x
    fitted_betas = beta_schedule(
        len(wave_data),
        change_day,
        beta_before,
        beta_after,
    )
    model = simulate_rk4(initial_state, dates, fitted_betas)

    model.insert(0, "wave", wave_key)
    model["actual_confirmed"] = actual
    model["data_section"] = np.where(
        np.arange(len(model)) < split_day,
        "fitting",
        "validation",
    )
    model["basic_R0"] = [
        basic_reproduction_number(beta) for beta in fitted_betas
    ]
    model["effective_Re"] = effective_reproduction_number(
        model,
        fitted_betas,
    )

    fitting_metrics = calculate_metrics(
        actual[:split_day],
        model.loc[: split_day - 1, "model_confirmed"],
    )
    validation_metrics = calculate_metrics(
        actual[split_day:],
        model.loc[split_day:, "model_confirmed"],
    )
    overall_metrics = calculate_metrics(
        actual,
        model["model_confirmed"],
    )

    metrics_rows = []
    for section, values in [
        ("fitting", fitting_metrics),
        ("validation", validation_metrics),
        ("overall", overall_metrics),
    ]:
        metrics_rows.append(
            {
                "wave": wave_key,
                "section": section,
                **values,
            }
        )

    information = {
        "wave": wave_key,
        "start_date": str(pd.Timestamp(dates.iloc[0]).date()),
        "end_date": str(pd.Timestamp(dates.iloc[-1]).date()),
        "fitting_end_date": str(
            pd.Timestamp(dates.iloc[split_day - 1]).date()
        ),
        "validation_start_date": str(
            pd.Timestamp(dates.iloc[split_day]).date()
        ),
        "beta_change_date": str(change_date.date()),
        "fitted_beta_before_peak": float(beta_before),
        "fitted_beta_after_peak": float(beta_after),
        "basic_R0_before_peak": basic_reproduction_number(beta_before),
        "basic_R0_after_peak": basic_reproduction_number(beta_after),
        "initial_S": float(initial_state[0]),
        "initial_E": float(initial_state[1]),
        "initial_I": float(initial_state[2]),
        "initial_R": float(initial_state[3]),
        "initial_recorded_confirmed": float(initial_state[4]),
    }

    return model, pd.DataFrame(metrics_rows), information


def fit_both_waves(data):
    """Fit and validate wave 1 and wave 2."""

    models = []
    metrics = []
    information = {}

    for wave_key in WAVES:
        model, wave_metrics, wave_information = fit_one_wave(
            data,
            wave_key,
        )
        models.append(model)
        metrics.append(wave_metrics)
        information[wave_key] = wave_information

    return (
        pd.concat(models, ignore_index=True),
        pd.concat(metrics, ignore_index=True),
        information,
    )


# -------------------------------------------------------------------
# 8. Basic and effective reproduction numbers
# -------------------------------------------------------------------

def basic_reproduction_number(beta):
    """Calculate R0 with the formula derived in the paper."""

    numerator = ETA * beta
    denominator = (ETA + MU) * (MU + DELTA + GAMMA)
    return float(numerator / denominator)


def effective_reproduction_number(model, beta_values):
    """Calculate Re(t) = R0(t) multiplied by S(t) / N(t)."""

    population = (
        model["model_S"]
        + model["model_E"]
        + model["model_I"]
        + model["model_R"]
    )
    susceptible_fraction = model["model_S"] / population.clip(lower=1)
    basic_values = np.array(
        [basic_reproduction_number(beta) for beta in beta_values]
    )
    return basic_values * susceptible_fraction.to_numpy()


# -------------------------------------------------------------------
# 9. Save files
# -------------------------------------------------------------------

def save_csv(table, file_path):
    """Save CSV without leaving a pandas file handle open."""

    try:
        file_path.write_text(
            table.to_csv(index=False),
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(
            "Could not save " + str(file_path)
            + ". Close it in Excel/VS Code and run again."
        ) from error


def make_parameter_table(fit_information):
    """Create one table containing paper and fitted parameters."""

    rows = [
        ["eta", ETA, "per day", "Reference paper"],
        ["gamma", GAMMA, "per day", "Reference paper"],
        ["delta", DELTA, "per day", "Reference paper"],
        ["recorded_portion_p", RECORDED_PORTION, "proportion", "Reference paper"],
        ["mu", MU, "per day", "Set to zero because paper gives no value"],
    ]

    for name, beta in BETA_SCENARIOS.items():
        rows.append(
            [
                "beta_" + name.lower().replace(" ", "_"),
                beta,
                "per day",
                "Reference paper scenario",
            ]
        )

    for wave_key, information in fit_information.items():
        rows.append(
            [
                wave_key + "_fitted_beta_before_peak",
                information["fitted_beta_before_peak"],
                "per day",
                "Least-squares fit",
            ]
        )
        rows.append(
            [
                wave_key + "_fitted_beta_after_peak",
                information["fitted_beta_after_peak"],
                "per day",
                "Least-squares fit",
            ]
        )

    return pd.DataFrame(
        rows,
        columns=["parameter", "value", "unit", "source"],
    )


# -------------------------------------------------------------------
# 10. Matplotlib plots
# -------------------------------------------------------------------

def finish_plot(fig, file_name, note, show_plots):
    """Add a simple note, save the plot, and optionally display it."""

    fig.text(0.02, 0.01, note, ha="left", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(PLOT_FOLDER / file_name, dpi=160)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_observed_waves(data, show_plots):
    """Plot the actual daily and active cases for both waves."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(data["date"], data["new_confirmed"], color="#0072B2")
    axes[0].set_title("Actual Daily Confirmed Cases")
    axes[0].set_ylabel("New cases")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["date"], data["infected_I"], color="#D55E00")
    axes[1].set_title("Actual Active Infected (I)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Active cases")
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        for wave in WAVES.values():
            axis.axvspan(
                pd.Timestamp(wave["start"]),
                pd.Timestamp(wave["end"]),
                alpha=0.10,
            )

    fig.suptitle("Nepal COVID-19: First and Second Waves")
    finish_plot(
        fig,
        "both_waves_actual.png",
        "The first wave has its largest daily rise in October 2020. "
        "The second wave is much larger and peaks in May 2021.",
        show_plots,
    )


def plot_removed(data, show_plots):
    """Show that observed R is recovered plus deaths."""

    fig, axis = plt.subplots(figsize=(12, 7))
    axis.plot(data["date"], data["recovered"], label="Recovered")
    axis.plot(data["date"], data["deaths"], label="Deaths")
    axis.plot(
        data["date"],
        data["removed_R"],
        label="R = recovered + deaths",
        linewidth=2,
    )
    axis.set_title("Removed Compartment R")
    axis.set_xlabel("Date")
    axis.set_ylabel("People")
    axis.legend()
    axis.grid(True, alpha=0.3)

    finish_plot(
        fig,
        "removed_is_recovered_plus_deaths.png",
        "Recovered people and deaths are both no longer actively infected, "
        "so this assignment combines them in R.",
        show_plots,
    )


def plot_beta_scenarios(data, scenarios, show_plots):
    """Plot the paper beta and three control reductions for each wave."""

    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]

    for wave_key, wave in WAVES.items():
        wave_data = get_wave_data(data, wave_key)
        selected = scenarios[scenarios["wave"] == wave_key]

        fig, axis = plt.subplots(figsize=(12, 7))
        axis.scatter(
            wave_data["date"],
            wave_data["confirmed"],
            label="Actual confirmed",
            color="black",
            s=10,
        )

        for color, (scenario_name, beta) in zip(
            colors,
            BETA_SCENARIOS.items(),
        ):
            scenario = selected[selected["scenario"] == scenario_name]
            axis.plot(
                scenario["date"],
                scenario["model_confirmed"],
                label=scenario_name + f", beta={beta:.4f}",
                color=color,
            )

        axis.set_title(wave["name"] + ": Paper Beta Scenarios")
        axis.set_xlabel("Date")
        axis.set_ylabel("Cumulative recorded cases")
        axis.legend()
        axis.grid(True, alpha=0.3)

        finish_plot(
            fig,
            wave_key + "_paper_beta_scenarios.png",
            "Lower beta means less transmission. The 25%, 50%, and 75% "
            "reductions represent stronger control measures.",
            show_plots,
        )


def plot_fitting_validation(models, fit_information, show_plots):
    """Plot actual cases against the fitted RK4 prediction."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for axis, (wave_key, wave) in zip(axes, WAVES.items()):
        model = models[models["wave"] == wave_key]
        information = fit_information[wave_key]

        axis.scatter(
            model["date"],
            model["actual_confirmed"],
            label="Actual confirmed",
            color="black",
            s=11,
        )
        axis.plot(
            model["date"],
            model["model_confirmed"],
            label="Fitted RK4 model",
            color="#0072B2",
            linewidth=2,
        )
        axis.axvline(
            pd.Timestamp(information["beta_change_date"]),
            color="#D55E00",
            linestyle="--",
            label="Beta change at case peak",
        )
        axis.axvline(
            pd.Timestamp(information["validation_start_date"]),
            color="#009E73",
            linestyle=":",
            label="Validation starts",
        )
        axis.set_title(wave["name"] + " Fitting and Validation")
        axis.set_xlabel("Date")
        axis.set_ylabel("Cumulative recorded cases")
        axis.legend()
        axis.grid(True, alpha=0.3)

    finish_plot(
        fig,
        "model_fitting_and_validation.png",
        "The first 80% of each wave fits beta. The final 20% is kept "
        "separate to check how well the fitted RK4 model predicts unseen dates.",
        show_plots,
    )


def plot_reproduction_numbers(models, show_plots):
    """Plot the fitted effective reproduction number for both waves."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    for axis, (wave_key, wave) in zip(axes, WAVES.items()):
        model = models[models["wave"] == wave_key]
        axis.plot(
            model["date"],
            model["basic_R0"],
            color="#009E73",
            linestyle=":",
            linewidth=2,
            label="Basic reproduction number R0",
        )
        axis.plot(
            model["date"],
            model["effective_Re"],
            color="#0072B2",
            linewidth=2,
            label="Effective reproduction number Re",
        )
        axis.axhline(
            1,
            color="#D55E00",
            linestyle="--",
            label="Threshold Re = 1",
        )
        axis.set_title(wave["name"])
        axis.set_xlabel("Date")
        axis.set_ylabel("Re")
        axis.legend()
        axis.grid(True, alpha=0.3)

    finish_plot(
        fig,
        "basic_and_effective_reproduction_numbers.png",
        "Re above 1 means infections can grow. Re below 1 means the "
        "outbreak tends to shrink. Re changes as beta and S/N change.",
        show_plots,
    )


def make_all_plots(data, scenarios, models, fit_information, show_plots):
    """Create every plot required by the assignment."""

    plot_observed_waves(data, show_plots)
    plot_removed(data, show_plots)
    plot_beta_scenarios(data, scenarios, show_plots)
    plot_fitting_validation(models, fit_information, show_plots)
    plot_reproduction_numbers(models, show_plots)


# -------------------------------------------------------------------
# 11. Findings and summary
# -------------------------------------------------------------------

def first_date_below_one(model, change_date):
    """Find when Re first becomes smaller than one after beta changes."""

    after_change = model[
        (model["date"] >= pd.Timestamp(change_date))
        & (model["effective_Re"] < 1)
    ]

    if after_change.empty:
        return "Re did not fall below 1 in this wave period"

    return str(pd.Timestamp(after_change.iloc[0]["date"]).date())


def make_summary(data, scenarios, models, metrics, fit_information):
    """Create a result-focused summary of every analysis and plot."""

    summary = {
        "task_completed": {
            "forward_RK4_for_both_waves": True,
            "paper_beta_scenarios": True,
            "model_fitting_and_validation": True,
            "basic_and_effective_reproduction_numbers": True,
            "notebook_created": False,
        },
        "reference_paper_values": {
            "beta": PAPER_BETA,
            "eta": ETA,
            "gamma": GAMMA,
            "delta": DELTA,
            "recorded_portion_p": RECORDED_PORTION,
            "paper_R0_at_beta_0_0536": basic_reproduction_number(PAPER_BETA),
            "beta_scenarios": BETA_SCENARIOS,
        },
        "important_model_meaning": {
            "S": "Susceptible people who can still become exposed.",
            "E": "Exposed people who are infected but are not yet infectious.",
            "I": "People who are infectious.",
            "R": "Recovered plus deaths, as required by the assignment.",
            "beta": "How quickly infection passes from I to S.",
            "R0": "Expected secondary cases when nearly everyone is susceptible.",
            "Re": "Expected secondary cases at a particular time after allowing for S/N.",
        },
        "wave_results": {},
        "paper_beta_scenario_results": {},
        "plot_findings": {
            "both_waves_actual.png": (
                "The second wave is much larger than the first wave. "
                "Daily confirmed cases peak in October 2020 for wave 1 "
                "and May 2021 for wave 2."
            ),
            "removed_is_recovered_plus_deaths.png": (
                "Recovered cases are the main part of R, while deaths are "
                "smaller but must still be included in R."
            ),
            "wave_1_paper_beta_scenarios.png": (
                "Reducing beta lowers the predicted cumulative cases. "
                "This agrees with the paper's control-strategy conclusion."
            ),
            "wave_2_paper_beta_scenarios.png": (
                "The paper's first-wave beta values predict much less growth "
                "than the actual second wave, so the same beta cannot explain both waves."
            ),
            "model_fitting_and_validation.png": (
                "A larger beta is needed before each peak and a smaller beta "
                "after the peak. This represents changing transmission and controls."
            ),
            "basic_and_effective_reproduction_numbers.png": (
                "Re above 1 supports growth; after beta falls and susceptibility "
                "decreases, Re moves toward or below 1 and the wave slows."
            ),
        },
    }

    for wave_key, wave in WAVES.items():
        wave_data = get_wave_data(data, wave_key)
        wave_model = models[models["wave"] == wave_key]
        information = fit_information[wave_key]
        validation = metrics[
            (metrics["wave"] == wave_key)
            & (metrics["section"] == "validation")
        ].iloc[0]

        peak = wave_data.loc[wave_data["new_confirmed"].idxmax()]

        summary["wave_results"][wave_key] = {
            "name": wave["name"],
            "date_start": wave["start"],
            "date_end": wave["end"],
            "highest_daily_confirmed_date": str(
                pd.Timestamp(peak["date"]).date()
            ),
            "highest_daily_confirmed_cases": float(peak["new_confirmed"]),
            "initial_values": {
                "S": information["initial_S"],
                "E": information["initial_E"],
                "I": information["initial_I"],
                "R": information["initial_R"],
            },
            "fitted_beta_before_peak": information[
                "fitted_beta_before_peak"
            ],
            "fitted_beta_after_peak": information[
                "fitted_beta_after_peak"
            ],
            "basic_R0_before_peak": information["basic_R0_before_peak"],
            "basic_R0_after_peak": information["basic_R0_after_peak"],
            "first_date_Re_below_1_after_beta_change": first_date_below_one(
                wave_model,
                information["beta_change_date"],
            ),
            "validation_RMSE_cases": float(validation["RMSE_cases"]),
            "validation_MAE_cases": float(validation["MAE_cases"]),
            "validation_MAPE_percent": float(validation["MAPE_percent"]),
            "validation_R_squared": float(validation["R_squared"]),
        }

        wave_scenarios = scenarios[scenarios["wave"] == wave_key]
        scenario_results = {}

        for scenario_name, beta in BETA_SCENARIOS.items():
            scenario = wave_scenarios[
                wave_scenarios["scenario"] == scenario_name
            ]
            scenario_results[scenario_name] = {
                "beta": beta,
                "basic_R0": basic_reproduction_number(beta),
                "predicted_confirmed_at_end": float(
                    scenario.iloc[-1]["model_confirmed"]
                ),
            }

        summary["paper_beta_scenario_results"][wave_key] = scenario_results

    wave_1_beta = fit_information["wave_1"]["fitted_beta_before_peak"]
    wave_2_beta = fit_information["wave_2"]["fitted_beta_before_peak"]

    summary["main_conclusions"] = [
        (
            "The second wave needed a much larger fitted transmission rate "
            f"({wave_2_beta:.4f}) than the first wave ({wave_1_beta:.4f})."
        ),
        (
            "The paper beta 0.0536 gives R0 below 1 with the paper formula. "
            "It can describe a controlled/declining period but not the rapid "
            "growth of Nepal's second wave."
        ),
        (
            "All 25%, 50%, and 75% beta reductions reduce predicted cases; "
            "stronger transmission control gives a smaller epidemic."
        ),
        (
            "The fitted beta falls after the case peak in both waves, which "
            "is consistent with reduced contact, behavior change, immunity, "
            "or public-health controls."
        ),
        (
            "Validation is not perfect because a simple SEIR model uses only "
            "two beta periods and cannot represent every reporting or behavior change."
        ),
    ]

    return summary


# -------------------------------------------------------------------
# 12. Run the complete assignment
# -------------------------------------------------------------------

def main(show_plots=True):
    """Run data preparation, RK4, fitting, validation, and plots."""

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    data = make_nepal_data()
    scenarios = run_paper_beta_scenarios(data)
    models, metrics, fit_information = fit_both_waves(data)
    parameters = make_parameter_table(fit_information)

    save_csv(data, DATA_FOLDER / "nepal_covid_two_waves.csv")
    save_csv(
        scenarios,
        OUTPUT_FOLDER / "paper_beta_scenarios_both_waves.csv",
    )
    save_csv(
        models,
        OUTPUT_FOLDER / "rk4_model_fitting_validation.csv",
    )
    save_csv(
        metrics,
        OUTPUT_FOLDER / "model_validation_metrics.csv",
    )
    save_csv(
        parameters,
        OUTPUT_FOLDER / "model_parameters.csv",
    )

    summary = make_summary(
        data,
        scenarios,
        models,
        metrics,
        fit_information,
    )
    (OUTPUT_FOLDER / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    make_all_plots(
        data,
        scenarios,
        models,
        fit_information,
        show_plots,
    )

    print("Done. Both-wave Nepal SEIR RK4 analysis is complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(show_plots=True)
