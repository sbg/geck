#!/usr/bin/env python

# This script reads a tab-delimited TXT file
# with the header 
#   feature1, [feature2, ...]
#   count,
# listing the aggregate counts in each category,
# and draws multivariate hypergeometric samples.
# (i.e. sub-samples of the multinomial)

import numpy as np
import argparse

def subsample_multinomial(counts, size_of_subsample):
    """
    draws sub-samples from multinomial counts

    :param counts: list of positive inters representing counts
    :param size_of_subsample: positive integer, smaller than sum of counts
    :return: sub-sampled counts: list of positive integers of the same length as counts
    """
    assert size_of_subsample <= np.sum(counts)
    all_samples = np.arange(1, np.sum(counts)+1, 1)
    sub_samples = np.sort(np.random.choice(all_samples, size_of_subsample, replace=False))
    
    bins = [0]
    csum = 0
    for count in counts:
        csum += count
        bins.append(csum)
    
    sub_counts = []
    for b in range(len(bins)-1):
        sub_count = len(sub_samples[(sub_samples > bins[b]) & (sub_samples <= bins[b+1])])
        sub_counts.append(sub_count)
    
    return sub_counts

parser = argparse.ArgumentParser()
parser.add_argument("input_txt", help="Tab-delimited TXT with aggregate counts. (FILE)")
parser.add_argument("subsample_size", help="Size of sub-sample to be drawn (INT)")
parser.add_argument("random_seed", help="Integer passed to numpy.random.seed() before sampling (INT)")
parser.add_argument("output_txt", help="Sub-sampled counts. (FILE)")
args = parser.parse_args()

labels = []
counts = []
with open(args.input_txt, 'r') as in_file:
    header = in_file.readline()[:-1].split('\t')
    assert 'count' in header, '"count" must be in header'
    count_idx = header.index('count')
    for line in in_file:
        record = line[:-1].split('\t')
        labels.append(tuple(record[:count_idx] +[None] +  record[count_idx+1:]))
        counts.append(int(record[count_idx]))

np.random.seed(int(args.random_seed))
sub_counts = subsample_multinomial(counts, int(args.subsample_size))

with open(args.output_txt, 'w') as out_file:
    out_file.write('\t'.join(header) + '\n')
    for label, sub_count in zip(labels, sub_counts):
        line = '\t'.join(label[:count_idx] + (str(sub_count),) + label[count_idx+1:]) + '\n'
        out_file.write(line)
