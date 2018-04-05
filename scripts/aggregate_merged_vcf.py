#!/usr/bin/env python

# This script counts number of variants 
# with each genotype combination across
# all samples in a merged vcf file.

# E.g. 
# if the input vcf contains the following genotypes,
# for a total of 4 variants
#     sample1  sample2  sample3
#     0|1      0|0      1|2
#     ./.      0|0      1|0
#     0|1      0|0      0|1
#     0|1      0|0      0|1

# then the output file will contain
#     sample1  sample2  sample3  count
#     0|1      0|0      1|2      1
#     ./.      0|0      1|0      1
#     0|1      0|0      0|1      2


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
import pysam
import argparse

def allele_idx_to_string(allele_index_tuple):
    gt = []
    for aidx in allele_index_tuple:
        if aidx is None:
            a = '.'
        else:
            a = str(aidx)
        gt.append(a)
    gt_str = ''.join(gt)
    return gt_str

parser = argparse.ArgumentParser()
parser.add_argument('input_vcf')
parser.add_argument('output_txt')
args = parser.parse_args()

input_path = args.input_vcf
output_path = args.output_txt

with pysam.VariantFile(input_path, 'r') as vcf:

    # get sample names
    samples = vcf.header.samples
    sample_strs = []
    for s in samples:
        sample_strs.append(str(s))
    
    # aggregate counts
    counts = {}
    for var in vcf.fetch():
        gt = tuple([var.samples[s].allele_indices for s in sample_strs])
        counts[gt] = counts[gt] + 1 if gt in counts else 1

# write output
gts = sorted(list(counts.keys()))
with open(output_path, 'w') as f_out:
    header = [s for s in sample_strs] + ['count']
    f_out.write('\t'.join(header) + '\n')
    for gt in gts:
        record = [allele_idx_to_string(gt[i]) for i in range(len(sample_strs))] + [str(counts[gt])] 
        f_out.write('\t'.join(record) + '\n')
