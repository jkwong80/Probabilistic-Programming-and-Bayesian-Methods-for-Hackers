
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


# I. Single Decay
# the livetime parameter
tau_parameter = 15.
decay_generator = pm.Exponential('decay_generator', 1/tau_parameter)

# let's generate some fake data
number_counts = 20
trigger_t = np.array([decay_generator.random().item() for _ in xrange(number_counts)])

# create plot showing the
bin_edges = np.linspace(0, 40, 100)
plt.figure()
plt.grid()
plt.hist(trigger_t, bins = bin_edges)
plt.xlabel('Time (sec)')
plt.ylabel('Count')


# create  the model
tau = pm.Exponential('tau', 1.0)
# decay_time = pm.Exponential('decay_time', 1/tau, value = trigger_t, observed = True, size = trigger_t.shape[0])
decay_time = pm.Exponential('decay_time', 1/tau, value = trigger_t, observed = True)

model = pm.Model([tau, decay_time])
mcmc = pm.MCMC(model)
mcmc.sample(10000)

# get the posterior distribution samples of tau
tau_samples = mcmc.trace('tau')[:]

burnin_instances = 5000

tau_samples_mean = tau_samples[burnin_instances:].mean()
tau_samples_std = tau_samples[burnin_instances:].std()


# plot the samples versus iteration
plt.figure()
plt.grid()
plt.plot(tau_samples)
plt.xlabel('Iteration')
# skip this many examples


# plot posterior distributions of tau
# We only plot after the

plt.figure()
plt.grid()
plt.hist(tau_samples[burnin_instances:], bins= 50, label = 'mean: %3.3f, std: %3.3f' %(tau_samples_mean, tau_samples_std))
plt.xlabel('tau (sec)')


# Create several fake samples of the best fit population
decay_time_realization = pm.Exponential('decay_time', 1 / tau_samples_mean)
plt.figure(figsize = [5, 8])
for i in xrange(4):
    plt.subplot(4, 1, i+1)
    plt.grid()

    plt.hist(np.array([decay_time_realization.random().item() for _ in xrange(number_counts)]), bins=bin_edges)
    plt.xlabel('Time (sec)')
    plt.ylabel('Count')
    plt.title('Realization #{} with tau = {}'.format(i, tau_samples_mean))



# lets try this out with various levels of statistics

number_counts_array = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])

number_sample_trials = 50

tau_samples_mean_matrix = np.zeros([number_sample_trials, len(number_counts_array)])
tau_samples_std_matrix = np.zeros([number_sample_trials, len(number_counts_array)])

decay_generator = pm.Exponential('decay_generator', 1/tau_parameter)

for number_counts_index, number_counts in enumerate(number_counts_array):
    for trial_num in xrange(number_sample_trials):

        # let's generate some fake data
        trigger_t = np.array([decay_generator.random().item() for _ in xrange(number_counts)])

        # create  the model
        tau = pm.Exponential('tau', 1.0)
        # decay_time = pm.Exponential('decay_time', 1/tau, value = trigger_t, observed = True, size = trigger_t.shape[0])
        decay_time = pm.Exponential('decay_time', 1/tau, value = trigger_t, observed = True)

        model = pm.Model([tau, decay_time])
        mcmc = pm.MCMC(model)
        mcmc.sample(10000)

        # get the posterior distribution samples of tau
        tau_samples = mcmc.trace('tau')[:]

        tau_samples_mean_matrix[trial_num, number_counts_index] = tau_samples[burnin_instances:].mean()
        tau_samples_std_matrix[trial_num, number_counts_index] = tau_samples[burnin_instances:].std()

# box plot of the MEAN of the samples of the tau posterior distribution

plt.figure()
plt.grid()
plt.boxplot(tau_samples_mean_matrix)
plt.xticks( np.arange(len(number_counts_array))+1, ['{}'.format(_) for _ in number_counts_array] )
plt.xlabel('Number of Counts')
plt.ylabel('Mean of Samples of the Posterior of tau')


# box plot of the STD of the samples of the tau posterior distribution

plt.figure()
plt.grid()

plt.boxplot(tau_samples_std_matrix)
plt.xticks( np.arange(len(number_counts_array))+1, ['{}'.format(_) for _ in number_counts_array] )
plt.xlabel('Number of Counts')
plt.ylabel('Std of Samples of the Posterior of tau')