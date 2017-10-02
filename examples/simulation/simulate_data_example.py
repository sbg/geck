#!/usr/bin/env python

# This is an example script showing how to use GeckModel
# to generate purely simulated variant counts,
# i.e. N[G1, G2], the aggregate number of variants
# whose trio genotype is called by G1 by tool 1 and  G2 by tool 2

import sys
import numpy
sys.path.append('../../geck')
from model import GeckModel
from data import GeckData

# initialize model
model = GeckModel()
family_genotypes = model.fg

# fix randomness
numpy.random.seed(12345)

# choose random model parameters
f_truth = model.random_f()
theta_truth = model.random_theta()
E_truth = model.random_e_array()

# specify the total number of variants
total_variants = int(1e6)

# generate sample
# N[G1, G1] is the (simulated) observed counts
# Ncomplete[G1,G2,g,m] is the (simulated) complete data,
# i.e. observed and hidden
N, Ncomplete = model.simulate_data(
    total_variants,
    f_truth,
    theta_truth,
    E_truth)

# load data from matrix format
data = GeckData()
data.load_matrix(['tool1', 'tool2'], N, family_genotypes)

# tab-delimited csv with header and one entry each line
data.save_data('./simulated_list.txt')

# tab-delimited matrix
data.save_data('./simulated_matrix.txt',
               format='matrix',
               sort_order=family_genotypes)
