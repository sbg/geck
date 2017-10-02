#!/usr/bin/env python

# This is an example script showing how to use GeckSolverGibbs
# to fit the GECK model to data using the Gibbs sampler

import sys
import numpy
sys.path.append('../../geck')
from data import GeckData
from solverGibbs import GeckSolverGibbs
from postprocess import GeckResults

tool_names = ['tool1', 'tool2']
data_file = './example_input.txt'

print 'Loading data'
data = GeckData()
data.load_file(tool_names, data_file)

print 'Initializing Gibbs sampler'
solver = GeckSolverGibbs()
solver.import_data(data)
print 'Running Gibbs sampler...'
numpy.random.seed(12345)
solver.run_sampling(burnin=5000,
                    every=100,
                    iterations=int(1e4),
                    verbose=True)
print 'Done'

print 'Analyzing results...'
results = GeckResults(tool_names, solver.n_array_complete_samples)

# e.g.
print 'Estimated joint confusion matrix, n[i,j,k] (father):'
print 'rows: GT called by tool1 (0/0, 0/1, 1/1)'
print 'cols: GT called by tool2 (0/0, 0/1, 1/1)'
n_avg = results.confusion_matrix('father').avg()
print 'true GT: 0/0'
print n_avg[0, :, :].astype(int)
print 'true GT: 0/1'
print n_avg[1, :, :].astype(int)
print 'true GT: 1/1'
print n_avg[2, :, :].astype(int)
print ''

# e.g.
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
