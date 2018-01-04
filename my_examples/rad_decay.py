import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

life_time = 56.
number_samples = 20000

decay_1_dist = pm.Exponential('decay_1', 1/life_time, size = number_samples)

decay_1_data = decay_1_dist.random()

number_bins = 200
bin_edges = np.arange(0,number_bins+1)
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.0

counts, binedges = np.histogram(decay_1_data, bin_edges)

tau = pm.DiscreteUniform('tau', 10, 100)
amp = pm.DiscreteUniform('amp', 10, 5000)

@pm.deterministic
def lambda_1(tau = tau, amp = amp):
    # out = np.zeros(number_bins)
    out = amp * np.exp(-bin_centers / tau)
    return(out)

observation = pm.Poisson("obs", lambda_1, value=counts, observed=True)

model = pm.Model([observation, amp, tau])


mcmc = pm.MCMC(model)
mcmc.sample(20000, 10000, 1)


tau_samples = mcmc.trace('tau')[:]
amp_samples = mcmc.trace('amp')[:]


plt.figure()
plt.subplot(2,1,1)
plt.grid()
plt.hist(tau_samples, bins = 200)

plt.subplot(2,1,2)
plt.grid()
plt.hist(amp_samples, bins = 200)


