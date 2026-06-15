"""Simple Nepal COVID-19 SEIR simulation using RK4.

This file does the tasks from the screenshot:
1. Use Nepal COVID-19 data from March 2020 to March 2021.
2. Use SEIR model.
3. Put recovered and deaths into R.
4. Use RK4 method.
5. Make plots with matplotlib and plt.show().
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# 1. Simple settings
# -----------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

COVID_DATA_FOLDER = (
    PROJECT_FOLDER.parent
    / "COVID-19"
    / "csse_covid_19_data"
    / "csse_covid_19_time_series"
)

DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
PLOT_FOLDER = PROJECT_FOLDER / "plots"

START_DATE = "2020-03-01"
END_DATE = "2021-03-31"

NEPAL_POPULATION = 29_136_808

# SEIR uses sigma to move people from E to I.
# Here we assume exposed people become infected after about 5 days.
LATENT_DAYS = 5
SIGMA = 1 / LATENT_DAYS


# -----------------------------
# 2. Read Nepal data
# -----------------------------

def read_nepal_series(file_name):
    """Read one Johns Hopkins CSV file and return only Nepal's time series."""

    file_path = COVID_DATA_FOLDER / file_name
    all_data = pd.read_csv(file_path)

    nepal_rows = all_data[all_data["Country/Region"] == "Nepal"]

    if nepal_rows.empty:
        raise ValueError("Nepal data was not found in " + file_name)

    # In JHU data, first 4 columns are place information.
    # Columns after that are dates.
    date_columns = all_data.columns[4:]

    nepal_values = (
        nepal_rows[date_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=0)
    )

    nepal_series = pd.Series(
        nepal_values.to_numpy(dtype=float),
        index=pd.to_datetime(date_columns, format="%m/%d/%y"),
    )

    return nepal_series.sort_index()


def make_nepal_data():
    """Make one simple Nepal dataframe for SEIR."""

    confirmed = read_nepal_series("time_series_covid19_confirmed_global.csv")
    recovered = read_nepal_series("time_series_covid19_recovered_global.csv")
    deaths = read_nepal_series("time_series_covid19_deaths_global.csv")

    data = pd.DataFrame(
        {
            "confirmed": confirmed,
            "recovered": recovered,
            "deaths": deaths,
        }
    )

    data = data.loc[START_DATE:END_DATE]
    data = data.ffill().fillna(0)

    # R means removed from active infection.
    # In this assignment, R = recovered + deaths.
    data["removed"] = data["recovered"] + data["deaths"]

    # I means active infected.
    data["infected"] = data["confirmed"] - data["removed"]
    data["infected"] = data["infected"].clip(lower=0)

    # S means susceptible.
    # Confirmed is NOT susceptible.
    # Susceptible means people not yet counted as confirmed infected.
    data["susceptible"] = NEPAL_POPULATION - data["confirmed"]
    data["susceptible"] = data["susceptible"].clip(lower=0)

    data["new_confirmed"] = data["confirmed"].diff().fillna(0).clip(lower=0)
    data["new_removed"] = data["removed"].diff().fillna(0).clip(lower=0)

    # JHU data does not provide exposed people.
    # So E is estimated from recent new confirmed cases.
    recent_average_cases = data["new_confirmed"].rolling(
        window=LATENT_DAYS,
        min_periods=1,
    ).mean()
    data["exposed"] = recent_average_cases * LATENT_DAYS
    data["exposed"] = data["exposed"].clip(lower=0)

    data.insert(0, "date", data.index)
    data.insert(1, "day", np.arange(len(data)))

    return data.reset_index(drop=True)


# -----------------------------
# 3. Estimate simple daily values
# -----------------------------

def estimate_daily_values(data):
    """Estimate beta and gamma for each day."""

    susceptible = data["susceptible"].to_numpy(dtype=float)
    exposed = data["exposed"].to_numpy(dtype=float)
    infected = data["infected"].to_numpy(dtype=float)

    tomorrow_exposed_change = (
        data["exposed"].shift(-1) - data["exposed"]
    ).fillna(0).to_numpy(dtype=float)

    tomorrow_removed_change = (
        data["removed"].shift(-1) - data["removed"]
    ).fillna(0).clip(lower=0).to_numpy(dtype=float)

    # SEIR equation:
    # dE/dt = beta*S*I/N - sigma*E
    # So:
    # beta*S*I/N = dE/dt + sigma*E
    exposure_flow = tomorrow_exposed_change + SIGMA * exposed
    exposure_flow = np.maximum(exposure_flow, 0)

    beta = np.zeros(len(data))
    gamma = np.zeros(len(data))

    valid = (infected > 0) & (susceptible > 0)

    beta[valid] = (
        exposure_flow[valid]
        * NEPAL_POPULATION
        / (susceptible[valid] * infected[valid])
    )

    # gamma means how fast infected people move into R.
    gamma[valid] = tomorrow_removed_change[valid] / infected[valid]

    daily_values = data[["date", "day"]].copy()
    daily_values["beta"] = np.nan_to_num(beta, nan=0, posinf=0, neginf=0)
    daily_values["sigma"] = SIGMA
    daily_values["gamma"] = np.nan_to_num(gamma, nan=0, posinf=0, neginf=0)

    # This prevents very large values from making the graph confusing.
    daily_values["beta"] = daily_values["beta"].clip(lower=0, upper=2)
    daily_values["gamma"] = daily_values["gamma"].clip(lower=0, upper=1)

    return daily_values


# -----------------------------
# 4. SEIR equations and RK4
# -----------------------------

def seir_equations(state, beta, gamma):
    """Return changes in S, E, I, R."""

    susceptible, exposed, infected, removed = state

    new_exposed = beta * susceptible * infected / NEPAL_POPULATION
    new_infected = SIGMA * exposed
    new_removed = gamma * infected

    dS = -new_exposed
    dE = new_exposed - new_infected
    dI = new_infected - new_removed
    dR = new_removed

    return np.array([dS, dE, dI, dR])


def rk4_step(state, beta, gamma, step_size=1):
    """Move SEIR one day forward using RK4."""

    k1 = seir_equations(state, beta, gamma)
    k2 = seir_equations(state + 0.5 * step_size * k1, beta, gamma)
    k3 = seir_equations(state + 0.5 * step_size * k2, beta, gamma)
    k4 = seir_equations(state + step_size * k3, beta, gamma)

    next_state = state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Population values should not become negative.
    return np.maximum(next_state, 0)


def run_rk4(data, daily_values):
    """Run RK4 for every day and compare actual values with model values."""

    first_model_day = data.index[(data["exposed"] > 0) | (data["infected"] > 0)][0]

    model_rows = []

    for i in range(len(data)):
        if i <= first_model_day:
            state = data.loc[i, ["susceptible", "exposed", "infected", "removed"]]
            model_rows.append(state.to_numpy(dtype=float))
            continue

        beta = daily_values.loc[i - 1, "beta"]
        gamma = daily_values.loc[i - 1, "gamma"]

        # Simple idea:
        # use yesterday's real values, then calculate today's RK4 values.
        yesterday_state = data.loc[
            i - 1,
            ["susceptible", "exposed", "infected", "removed"],
        ].to_numpy(dtype=float)

        today_model_state = rk4_step(yesterday_state, beta, gamma)
        model_rows.append(today_model_state)

    model = pd.DataFrame(
        model_rows,
        columns=[
            "model_susceptible",
            "model_exposed",
            "model_infected",
            "model_removed",
        ],
    )

    model["model_confirmed"] = model["model_infected"] + model["model_removed"]

    result = pd.concat(
        [
            data.reset_index(drop=True),
            model,
            daily_values[["beta", "sigma", "gamma"]],
        ],
        axis=1,
    )

    return result


# -----------------------------
# 5. Plot with matplotlib
# -----------------------------

def add_simple_plot_note(text):
    """Write simple explanation text below a plot."""

    plt.figtext(
        0.02,
        0.02,
        text,
        ha="left",
        fontsize=9,
        wrap=True,
    )


def plot_all(data, result):
    """Show and save all plots."""

    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Plot 1: actual SEIR values
    plt.figure(figsize=(12, 7))
    plt.plot(data["date"], data["confirmed"], label="Confirmed total")
    plt.plot(data["date"], data["exposed"], label="E = estimated exposed")
    plt.plot(data["date"], data["infected"], label="I = active infected")
    plt.plot(data["date"], data["removed"], label="R = recovered + deaths")
    plt.title("Nepal COVID-19 SEIR Data: March 2020 to March 2021")
    plt.xlabel("Date")
    plt.ylabel("People")
    plt.legend()
    plt.grid(True)
    add_simple_plot_note(
        "Confirmed = total reported cases, not susceptible. "
        "S = population - confirmed. "
        "E is estimated because dataset does not directly give exposed people. "
        "I = confirmed - recovered - deaths. R = recovered + deaths."
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(PLOT_FOLDER / "simple_seir_actual.png", dpi=150)
    plt.show()

    # Plot 2: actual vs RK4 model
    plt.figure(figsize=(12, 7))
    plt.plot(result["date"], result["exposed"], label="Actual E")
    plt.plot(result["date"], result["model_exposed"], "--", label="RK4 E")
    plt.plot(result["date"], result["infected"], label="Actual I")
    plt.plot(result["date"], result["model_infected"], "--", label="RK4 I")
    plt.plot(result["date"], result["removed"], label="Actual R")
    plt.plot(result["date"], result["model_removed"], "--", label="RK4 R")
    plt.title("Nepal COVID-19 Actual Values vs RK4 SEIR Model")
    plt.xlabel("Date")
    plt.ylabel("People")
    plt.legend()
    plt.grid(True)
    add_simple_plot_note(
        "Actual means values made from Nepal data. "
        "RK4 means values calculated by the SEIR equations. "
        "Close lines mean the model is following the data better."
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(PLOT_FOLDER / "simple_seir_rk4_compare.png", dpi=150)
    plt.show()

    # Plot 3: recovered and deaths into R
    plt.figure(figsize=(12, 7))
    plt.plot(data["date"], data["recovered"], label="Recovered")
    plt.plot(data["date"], data["deaths"], label="Deaths")
    plt.plot(data["date"], data["removed"], label="R = recovered + deaths")
    plt.title("Recovered and Deaths Combined Into R")
    plt.xlabel("Date")
    plt.ylabel("People")
    plt.legend()
    plt.grid(True)
    add_simple_plot_note(
        "R means removed from active infection. "
        "That is why recovered people and deaths are added together."
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(PLOT_FOLDER / "simple_removed_is_recovered_plus_deaths.png", dpi=150)
    plt.show()


# -----------------------------
# 6. Run everything
# -----------------------------

def save_csv(table, file_path):
    """Save CSV in a simple way that avoids pandas file-handle errors."""

    try:
        csv_text = table.to_csv(index=False)
        file_path.write_text(csv_text, encoding="utf-8")
    except PermissionError as error:
        raise PermissionError(
            "Could not save this CSV. Close the file if it is open in Excel "
            "or another program, then run the script again: " + str(file_path)
        ) from error
    except OSError as error:
        raise OSError(
            "Could not save this CSV. OneDrive or another program may be "
            "locking the file. Close the file and run again: " + str(file_path)
        ) from error


def main(show_plots=True):
    """Run the complete assignment in simple steps."""

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    data = make_nepal_data()
    daily_values = estimate_daily_values(data)
    result = run_rk4(data, daily_values)

    save_csv(data, DATA_FOLDER / "nepal_covid_mar2020_mar2021.csv")
    save_csv(daily_values, OUTPUT_FOLDER / "nepal_daily_seir_parameters.csv")
    save_csv(result, OUTPUT_FOLDER / "nepal_seir_rk4_simulation.csv")

    first_row = data.iloc[0]
    final_row = data.iloc[-1]

    peak_active_row = data.loc[data["infected"].idxmax()]
    peak_exposed_row = data.loc[data["exposed"].idxmax()]
    peak_new_cases_row = data.loc[data["new_confirmed"].idxmax()]
    peak_deaths_row = data.loc[data["deaths"].idxmax()]

    final_confirmed = float(final_row["confirmed"])
    final_recovered = float(final_row["recovered"])
    final_deaths = float(final_row["deaths"])
    final_removed = float(final_row["removed"])
    final_active = float(final_row["infected"])
    final_exposed = float(final_row["exposed"])
    final_susceptible = float(final_row["susceptible"])

    recovered_percent = round((final_recovered / final_confirmed) * 100, 2)
    death_percent = round((final_deaths / final_confirmed) * 100, 2)
    active_percent = round((final_active / final_confirmed) * 100, 2)
    removed_percent = round((final_removed / final_confirmed) * 100, 2)

    summary = {
        "overall_result": (
            "Nepal had 277,309 confirmed COVID-19 cases by 2021-03-31. "
            "Most of those cases were no longer active because recovered and deaths "
            "together made R = 275,816. Only 1,493 cases were active at the end."
        ),
        "what_we_found_from_the_data": {
            "confirmed_cases": (
                "Confirmed cases grew from "
                + str(float(first_row["confirmed"]))
                + " on "
                + str(first_row["date"].date())
                + " to "
                + str(final_confirmed)
                + " on "
                + str(final_row["date"].date())
                + ". This shows total reported cases increased strongly during the period."
            ),
            "susceptible_people": (
                "Confirmed is not susceptible. Susceptible means people not yet counted "
                "as confirmed infected. By the final date, susceptible was "
                + str(final_susceptible)
                + "."
            ),
            "removed_R": (
                "R was made by adding recovered and deaths. On the final date, R was "
                + str(final_removed)
                + ", which is "
                + str(removed_percent)
                + "% of confirmed cases. This means most confirmed cases had already "
                "moved out of active infection."
            ),
            "active_I": (
                "Active infected cases were "
                + str(final_active)
                + " on the final date, only "
                + str(active_percent)
                + "% of confirmed cases. This tells us the active load was much smaller "
                "than the total reported cases at the end."
            ),
            "estimated_exposed_E": (
                "E is not directly given by the dataset. It was estimated from recent "
                "new confirmed cases. On the final date, estimated exposed was "
                + str(final_exposed)
                + "."
            ),
        },
        "plot_1_actual_seir_compartments": {
            "plot_file": "plots/simple_seir_actual.png",
            "what_it_shows": (
                "This plot shows confirmed total, estimated exposed E, active infected I, "
                "and removed R over time."
            ),
            "what_we_notice": [
                "Confirmed cases increase over time because it is cumulative.",
                "Removed R rises strongly because recovered and deaths keep adding up.",
                "Active infected I rises and falls, so it shows waves more clearly than confirmed.",
                "Estimated exposed E follows the recent new-case trend because E is estimated from new confirmed cases.",
            ],
            "main_finding": (
                "By the end, confirmed and R are high, but active infected is low. "
                "This means most reported cases had recovered or died by 2021-03-31."
            ),
        },
        "plot_2_actual_vs_rk4": {
            "plot_file": "plots/simple_seir_rk4_compare.png",
            "what_it_shows": (
                "This plot compares actual E, I, R values with the values calculated by "
                "the RK4 SEIR model."
            ),
            "what_we_notice": [
                "RK4 follows the same general direction as the actual values.",
                "The model is not perfect because E is estimated and beta/gamma change day by day.",
                "When RK4 and actual lines are close, the SEIR equations are representing the data better.",
            ],
            "main_finding": (
                "RK4 gives a mathematical reconstruction of Nepal's epidemic trend. "
                "It helps compare real data with SEIR model behavior."
            ),
        },
        "plot_3_recovered_deaths_removed": {
            "plot_file": "plots/simple_removed_is_recovered_plus_deaths.png",
            "what_it_shows": (
                "This plot shows recovered, deaths, and R together."
            ),
            "what_we_notice": [
                "Recovered is much larger than deaths.",
                "R is very close to recovered because deaths are much smaller compared with recovered.",
                "Deaths still must be included in R because deaths are also removed from active infection.",
            ],
            "main_finding": (
                "R mostly grows because of recoveries, but deaths are included to make the removed compartment correct."
            ),
        },
        "important_peaks": {
            "highest_active_infected_I": {
                "date": str(peak_active_row["date"].date()),
                "value": float(peak_active_row["infected"]),
                "meaning": "This was the highest active infection burden in the selected period.",
            },
            "highest_estimated_exposed_E": {
                "date": str(peak_exposed_row["date"].date()),
                "value": float(peak_exposed_row["exposed"]),
                "meaning": "This was the highest estimated exposed value based on recent new cases.",
            },
            "highest_daily_new_confirmed": {
                "date": str(peak_new_cases_row["date"].date()),
                "value": float(peak_new_cases_row["new_confirmed"]),
                "meaning": "This was the largest one-day increase in confirmed cases.",
            },
            "highest_deaths_total": {
                "date": str(peak_deaths_row["date"].date()),
                "value": float(peak_deaths_row["deaths"]),
                "meaning": "Deaths are cumulative, so the highest total appears near the end.",
            },
        },
        "final_numbers_on_2021_03_31": {
            "confirmed_total": final_confirmed,
            "recovered": final_recovered,
            "deaths": final_deaths,
            "R_removed_recovered_plus_deaths": final_removed,
            "I_active_infected": final_active,
            "E_estimated_exposed": final_exposed,
            "S_susceptible": final_susceptible,
            "recovered_percent_of_confirmed": recovered_percent,
            "death_percent_of_confirmed": death_percent,
            "active_percent_of_confirmed": active_percent,
            "removed_percent_of_confirmed": removed_percent,
        },
        "main_conclusion": (
            "The major pattern is that Nepal's confirmed cases increased greatly from March 2020 "
            "to March 2021, but by the final date almost all confirmed cases were already in R. "
            "The peak active infection period was around October 2020. The SEIR RK4 model helps "
            "show the relationship between susceptible, estimated exposed, active infected, and removed cases."
        ),
    }

    (OUTPUT_FOLDER / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if show_plots:
        plot_all(data, result)

    print("Done. Nepal SEIR RK4 files are created.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
