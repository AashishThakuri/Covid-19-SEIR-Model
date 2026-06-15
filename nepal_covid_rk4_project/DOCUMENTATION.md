# Simple Explanation

The file `nepal_covid_rk4.py` is now written in a simple style.

## What The Code Does

1. Reads Johns Hopkins COVID-19 CSV files.
2. Filters only Nepal.
3. Keeps data from March 1, 2020 to March 31, 2021.
4. Builds SEIR values:
   - `S = susceptible`
   - `E = exposed`
   - `I = infected`
   - `R = removed`
5. Uses:

```text
R = recovered + deaths
```

6. Estimates `E` because the dataset does not directly give exposed people.
7. Uses RK4 to calculate model values.
8. Saves CSV outputs.
9. Shows plots using `plt.show()`.

## Simple Meaning

### Confirmed

`confirmed` means total reported COVID cases so far.

It is not susceptible.

### Susceptible

`susceptible` means people not yet counted as confirmed infected.

```text
susceptible = Nepal population - confirmed
```

### Exposed

`exposed` means people who may have been infected but are not yet active infected.

The dataset does not directly provide this, so the code estimates it from recent new confirmed cases.

### Infected

`infected` means active infected cases.

```text
infected = confirmed - recovered - deaths
```

### Removed

`removed` means people removed from active infection.

```text
removed = recovered + deaths
```

## SEIR Equations

```text
dS/dt = - beta * S * I / N
dE/dt =   beta * S * I / N - sigma * E
dI/dt =   sigma * E - gamma * I
dR/dt =   gamma * I
```

## RK4

RK4 means fourth-order Runge-Kutta.

It calculates the next day using four slope estimates:

```text
k1, k2, k3, k4
```

Then it combines them to get a better estimate than a very simple one-step method.
