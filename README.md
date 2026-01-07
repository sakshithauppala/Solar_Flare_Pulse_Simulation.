# Solar_Flare_Pulse_Simulation.
pip install numpy pandas matplotlib
python main.py

# The Solar Flare Pulse: Stochastic Signal Recovery

## Project Description
[cite_start]This project performs Bayesian Parameter Estimation on a localized magnetic reconnection event (Stellar Flare)[cite: 6]. [cite_start]It uses sensor data from the Solar Dynamics Observatory (SDO) to predict the energetics of the event[cite: 6, 8].

## Mathematical Model
[cite_start]The intensity $S(t)$ follows an exponential growth and rapid quenching model[cite: 7]:
[cite_start]$$S(t) = A e^{\{1 - \tanh[2(t - \tau)]\}} \sin(\omega t)$$ 

## Methodology
- [cite_start]**Algorithm**: Metropolis-Hastings MCMC (Random Walk).
- [cite_start]**Likelihood**: Log-Likelihood assuming a 20% relative error per data point[cite: 23, 24].
- [cite_start]**Priors**: Uniform priors for $A \in (0,2)$, $\tau \in (1,10)$, and $\omega \in (1,20)$[cite: 22].

## Requirements
- Python 3.x
- NumPy, Pandas, Matplotlib

## How to Run
1. Ensure `flare_data.csv` is in the directory.
2. Execute `python main.py`.
3. Results (Trace plots and Histograms) will be displayed and saved as PNG files.
