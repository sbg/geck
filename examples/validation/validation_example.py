#!/usr/bin/env python

# This script
#  1. runs GECK's Gibbs solver on validation data,
#     estimates metrics (from N_validation_data.txt)
#  2. calculates the true metrics (from Ncomplete_validation_data.txt)
#  3. prints them to stdout for comparison


# geck: Genotype Error Comparator Kit, for benchmarking genotyping tools
# Copyright (C) 2017 Seven Bridges Genomics Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
import numpy
sys.path.append('../../geck')
from data import GeckData
from model import GeckModel
from solverGibbs import GeckSolverGibbs
from postprocess import GeckResults


##############################################
#  1.  Estimate benchmarking metrics with GECK
##############################################

tool_names = ['bwa+gatk', 'gral+hc']
data_file = './N_validation_data.txt'

print 'Loading data'
data = GeckData()
data.load_file(tool_names, data_file)

print 'Initializing Gibbs sampler'
solver = GeckSolverGibbs()
solver.import_data(data)
solver.n_array[0, 0] = 52054804
# set number of unobserved "variants" to size of dbSNP

print 'Running Gibbs sampler...'
numpy.random.seed(12345)
solver.run_sampling(burnin=5000,
                    every=100,
                    iterations=int(1e4),
                    verbose=True)
print 'Done'

print 'Analyzing results...'
person = 'family'  # could be any of ['father', 'mother', 'child', 'family']
mode = 'hard'      # could be any of ['soft', 'hard', '00', '01', '11']

results = GeckResults(tool_names, solver.n_array_complete_samples)
n_avg = results.confusion_matrix(person).avg()

F1_avg = results.get_f_score(person)[mode]['tool1'].avg()
F1_low = results.get_f_score(person)[mode]['tool1'].percentile(0.05)
F1_high = results.get_f_score(person)[mode]['tool1'].percentile(0.95)

F2_avg = results.get_f_score(person)[mode]['tool2'].avg()
F2_low = results.get_f_score(person)[mode]['tool2'].percentile(0.05)
F2_high = results.get_f_score(person)[mode]['tool2'].percentile(0.95)

F_delta_avg = results.get_f_score(person)[mode]['diff'].avg()
F_delta_low = results.get_f_score(person)[mode]['diff'].percentile(0.05)
F_delta_high = results.get_f_score(person)[mode]['diff'].percentile(0.95)


#########################################
#  2. Calculate true benchmarking metrics
#########################################

def load_Ncomplete(fullpath, model):
    lenvfg = len(model.vfg)
    lenfg = len(model.fg)
    Ncomplete = numpy.zeros((lenvfg, lenfg, lenfg))
    with open(fullpath, 'r') as f:
        header = f.readline()
        assert header[:-1] == \
            '\t'.join(["fatherTrue", "motherTrue", "childTrue",
                       "father1", "mother1", "child1",
                       "father2", "mother2", "child2",
                       "count"])
        for line in f:
            record = line[:-1].split('\t')
            g = tuple(record[0:3])
            G1 = tuple(record[3:6])
            G2 = tuple(record[6:9])
            count = int(record[9])

            g_idx = model.vfg.index(g)
            G1_idx = model.fg.index(G1)
            G2_idx = model.fg.index(G2)
            Ncomplete[g_idx, G1_idx, G2_idx] += count
    return Ncomplete

Ncomplete_fullpath = './Ncomplete_validation_data.txt'
model = GeckModel()
K_dict = {'father': model.k_array_father,
          'mother': model.k_array_mother,
          'child': model.k_array_child,
          'family': model.k_array}
result_dummy = GeckResults([], [numpy.zeros((27, 27, 15, 15))])
# only needed because its methods are instance-bound

Ncomplete = load_Ncomplete(Ncomplete_fullpath, model)
n_truth = numpy.einsum('gGH,ijkgGH->ijk', Ncomplete, K_dict[person])
n1 = numpy.einsum('ijk->ij', n_truth)
n2 = numpy.einsum('ijk->ik', n_truth)

F1_truth = result_dummy._calculate_f_score(n1, mode)
F2_truth = result_dummy._calculate_f_score(n2, mode)
F_delta_truth = F2_truth - F1_truth


#############
#  3. Compare
#############

print 'Estimated joint confusion matrix, n[i,j,k] (' + str(person) + '):'
print 'rows: GT called by {} (0/0, 0/1, 1/1)'.format(tool_names[0])
print 'cols: GT called by {} (0/0, 0/1, 1/1)'.format(tool_names[1])
print 'true GT: 0/0'
print n_avg[0, :, :].astype(int)
print 'true GT: 0/1'
print n_avg[1, :, :].astype(int)
print 'true GT: 1/1'
print n_avg[2, :, :].astype(int)
print ''
print 'True joint confusion matrix, n[i,j,k] (' + str(person) + '):'
print 'rows: GT called by {} (0/0, 0/1, 1/1)'.format(tool_names[0])
print 'cols: GT called by {} (0/0, 0/1, 1/1)'.format(tool_names[1])
print 'true GT: 0/0'
print n_truth[0, :, :].astype(int)
print 'true GT: 0/1'
print n_truth[1, :, :].astype(int)
print 'true GT: 1/1'
print n_truth[2, :, :].astype(int)
print ''


# Compare estimated F-scores with true F-scores
print 'Estimated F score (' + str(mode) + ')' + \
    ' (+ 5th, 95th percentiles) (' + str(person) + '):'
print tool_names[0] + ': ' + str(F1_avg) + \
    ' (' + str(F1_low) + ', ' + str(F1_high) + ')'
print tool_names[1] + ': ' + str(F2_avg) + \
    ' (' + str(F2_low) + ', ' + str(F2_high) + ')'
print 'diff: ' + str(F_delta_avg) + \
    ' (' + str(F_delta_low) + ', ' + str(F_delta_high) + ')'
print ''
print 'True F score (' + str(mode) + ') ' + '(' + str(person) + '):'
print tool_names[0] + ': ' + str(F1_truth)
print tool_names[1] + ': ' + str(F2_truth)
print 'diff: ' + str(F_delta_truth)
print ''


# Full report of estimated values
print results.report_metrics(mode='all',
                             person='all',
                             percentiles=(0.05, 0.95))
