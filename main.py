import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
try:
    data = pd.read_csv('flare_data.csv')
    t_data = data['t'].values
    y_data = data['y_data'].values
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 2. Define the Analytical Model 
def model_s(t, A, tau, omega):
    # S(t) = A * e^{1 - tanh[2(t - tau)]} * sin(omega * t)
    return A * np.exp(1 - np.tanh(2 * (t - tau))) * np.sin(omega * t)

# 3. Statistical Logic [cite: 22, 26, 28]
def log_prior(theta):
    A, tau, omega = theta
    # Ranges specified in problem statement [cite: 13, 14, 15]
    if 0 < A < 2 and 1 < tau < 10 and 1 < omega < 20:
        return 0.0
    return -np.inf

def log_likelihood(theta, t, y):
    A, tau, omega = theta
    y_model = model_s(t, A, tau, omega)
    # sigma_i = 0.2 * |y_data,i| [cite: 28]
    sigma = 0.2 * np.abs(y)
    sigma = np.where(sigma == 0, 1e-6, sigma) # Numerical stability [cite: 30]
    return -np.sum(((y - y_model)**2) / (sigma**2))

def log_posterior(theta, t, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y)

# 4. Metropolis-Hastings MCMC 
def run_mcmc(iterations=10000):
    current_theta = np.array([1.0, 5.0, 10.0]) # Starting point
    samples = []
    
    for i in range(iterations):
        # Propose new step (Random Walk) 
        proposal = current_theta + np.random.normal(0, 0.05, size=3)
        
        # Accept/Reject logic
        p_accept = log_posterior(proposal, t_data, y_data) - log_posterior(current_theta, t_data, y_data)
        
        if np.log(np.random.rand()) < p_accept:
            current_theta = proposal
            
        samples.append(current_theta)
    
    return np.array(samples)

# 5. Execution and Deliverables [cite: 33, 35, 36]
samples = run_mcmc(15000)
burn_in = 5000
clean_samples = samples[burn_in:]

# Generate Trace Plots [cite: 33]
labels = ['A', 'tau', 'omega']
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axes[i].plot(samples[:, i])
    axes[i].set_ylabel(labels[i])
plt.tight_layout()
plt.savefig('trace_plots.png')
plt.show()

# Generate Histograms [cite: 35]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    axes[i].hist(clean_samples[:, i], bins=30, color='orange', edgecolor='black')
    axes[i].set_title(f'Posterior: {labels[i]}')
plt.tight_layout()
plt.savefig('posterior_dist.png')
plt.show()

# Best Fit MAP Estimates [cite: 36]
map_values = [np.mean(clean_samples[:, i]) for i in range(3)]
print(f"--- MAP Estimates ---")
print(f"A: {map_values[0]:.4f}, tau: {map_values[1]:.4f}, omega: {map_values[2]:.4f}")
