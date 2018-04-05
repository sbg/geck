#!/usr/bin/env python

# This script takes "observed" GT combination counts
# aka N.txt
# It imputes a user-defined number in place of N[0,0] 
# and fits the GECK model with Gibbs solver
# and produces two outputs
#  1. tar file containig n_... and metrics_... files
#  2. results.json containing the metrics


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

from __future__ import unicode_literals
from io import open
import sys
import numpy
import argparse
import json
import subprocess
import itertools

sys.path.append('/geck/')
from data import GeckData
from model import GeckModel
from solverGibbs import GeckSolverGibbs
from postprocess import GeckResults


parser = argparse.ArgumentParser()
parser.add_argument("N_txt", 
                    help="N (observed data) counts.")
parser.add_argument("unobserved_variants", 
                    help="number of variants that none of the two pipelines detected (INT)")
parser.add_argument("tool_names", 
                    help="name of the two tools", 
                    nargs=2)
parser.add_argument("random_seed", 
                    help="integer passed to numpy.random.seed() before Gibbs sampling starts (INT)")
parser.add_argument("burnin", 
                    help="number of iterations dropped at the beginning of Gibbs sampling (INT)")
parser.add_argument("total_iterations", 
                    help="number of Gibb sampling iterations after burn-in period (INT)")
parser.add_argument("sampling_period", 
                    help="period of recording samples from Gibbs sampler (INT)")
parser.add_argument("metrics_json",
                    help="filename for JSON output of the benchmarking estimated metrics")
parser.add_argument("percentiles_to_report", 
                    help="list of FLOAT (0<, <1), representing the percentiles to report in the output",
                    nargs='*')
args = parser.parse_args()

# load data
data = GeckData()
data.load_file(args.tool_names, args.N_txt)

# run Gibb sampling
solver = GeckSolverGibbs()
solver.import_data(data)
solver.n_array[0,0] = int(args.unobserved_variants)
numpy.random.seed(int(args.random_seed))
solver.run_sampling(burnin=int(args.burnin),
                   every=int(args.sampling_period),
                   iterations=int(args.total_iterations),
                   verbose=True)

# analyze results
results = GeckResults(args.tool_names, solver.n_array_complete_samples)
metrics_dict = results.dict_metrics(percentiles=tuple(percentiles))

# write json
with open(args.metrics_json, 'w') as out_file:
    out_file.write(json.dumps(metrics_dict, sort_keys=True, indent=4, separators=(',', ': ')))

# write n samples
n_labels = []
model = GeckModel()
for i in model.ig:
    for j in model.ig:
        for k in model.ig:
            n_labels.append(str(i) + '->' + str(j) + ',' + str(k))
n_header = '\t'.join(n_labels) + '\n'
n_files = []
for person in ['father', 'mother', 'child', 'family']:
    n_file = 'n_' + person + '.txt'
    n_files.append(n_file)
    with open(n_file, 'w') as out_file:
        out_file.write(n_header)
        for n in results.confusion_matrix(person).samples:
            out_file.write('\t'.join([str(int(nijk)) for nijk in n.flatten()]) + '\n')

# write metrics samples
metrics = ["precision", "recall", "Fscore"]
types = ["tool1", "tool2", "diff"]
modes = ["hard", "soft", "00", "01", "11"]
metrics_methods = {
    "precision": results._calculate_precision,
    "recall": results._calculate_recall,
    "Fscore": results._calculate_f_score
}
results_methods = {
    "precision": results.get_precision,
    "recall": results.get_recall,
    "Fscore": results.get_f_score
}
metrics_labels = []
for metric, typ, mode in itertools.product(metrics, types, modes):
    metrics_labels.append('_'.join([metric, typ, mode]))
metrics_header = '\t'.join(metrics_labels) + '\n'
metrics_files = []
for person in ['father', 'mother', 'child', 'family']:
    metrics_file = 'metrics_' + person + '.txt'
    metrics_files.append(metrics_file)
    with open(metrics_file, 'w') as out_file:
        out_file.write(metrics_header)
        number_of_samples = len(results.confusion_matrix(person).samples)
        for i in range(number_of_samples):
            record = []
            for metric, typ, mode in itertools.product(metrics, types, modes):
                method = results_methods[metric]
                prf = method(person)[mode][typ].samples[i]
                record.append(str(prf))
            out_file.write('\t'.join(record) + '\n')
