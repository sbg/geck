#!/usr/bin/env python

# This is an example script showing how to use GeckSolverEM
# to fit the GECK model to data using the Exepecation Maximization solver

import sys
sys.path.append('../../geck')
from data import GeckData
from solverEM import GeckSolverEM
from postprocess import GeckResults


# Load data

# load aggregate variant counts, N, from file
# where N[G1, G2] is the aggregate number of variants
# whose trio genotype is called by G1 by tool 1 and  G2 by tool 2
tools = ['tool1', 'tool2']
data_file = './example_input.txt'

data = GeckData()
data.load_file(tools, data_file)


# Run Expectation maximization

# initialize solver
solver = GeckSolverEM()

# set initial values
solver.init_f(f0=solver.random_f())
solver.init_theta(theta0=solver.random_theta())
solver.init_e_array(e0=solver.random_e_array())

# regularization limits
solver.f_min = 1e-5
solver.theta_min = 1e-5
solver.e_min = 1e-5

# load data from GeckData instance
solver.import_data(data)

# perform EM iterations to find the Maximum Likelihood point
# (this updates the f, theta, E and R variables of solver)
print 'Running EM...'
solver.fit(threshold=1e-8, iter_max=int(1e4), report_every=10, verbose=True)
print 'Done'

# draw samples for Ncomplete using the fitted model
# (this populates solver.Ncomplete_samples with samples)
print 'Sampling model...'
iterations = solver.sample_n_array_complete(100)
print 'Done'

# Analyze results
print 'Analyzing results...'
results = GeckResults(tools, solver.n_array_complete_samples)

# Joint confusion matrix (for father's variants):
# n[i,j,k] is the estimated number of variants with true genotype i
# that are called j by tool1 and k by tool2
# (i,j,k are from ["00", "01", "11"])
n_avg = results.confusion_matrix('father').avg()
print 'Estimated joint confusion matrix, n[i,j,k] (father):'
print 'rows: GT called by tool1 (0/0, 0/1, 1/1)'
print 'cols: GT called by tool2 (0/0, 0/1, 1/1)'
print 'true GT: 0/0'
print n_avg[0, :, :].astype(int)
print 'true GT: 0/1'
print n_avg[1, :, :].astype(int)
print 'true GT: 1/1'
print n_avg[2, :, :].astype(int)
print ''

# Precision, Recall, F-score
# (here only showing the "hard" version of them
#  where "1/1" <-> "0/1" mix-ups are not excused)

# Fscore improvement (of mother's variants)
print 'F score (hard) difference (+ 5th, 95th percentiles) (mother)'
F_delta = results.get_f_score('mother')['hard']['diff'].avg()
F_delta_low = results.get_f_score('mother')['hard']['diff'].percentile(0.05)
F_delta_high = results.get_f_score('mother')['hard']['diff'].percentile(0.95)
print str(F_delta) + ' (' + str(F_delta_low) + ', ' + str(F_delta_high) + ')'
print ''

# Full report
print results.report_metrics(mode='all',
                             person='all',
                             percentiles=(0.05, 0.95))
