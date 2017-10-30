#!/usr/bin/env python

# This script sanitizes the genotype combinations counts:
#  - removes multiallelic counts
#  - disregards phasing (aggregates over them)
#  - assumes reference allele for missing genotypes (i.e. '.' == '0')

# Inputs:
# -------
#  * tool1: name of first variant determination pipeline
#  * tool2: name of second variant determination pipeline
#  * a tab-separated file listing the aggregate counts
#    of different genotype combinations in a 6-sample vcf, and
#  * a pedigree file, storing the relationships of the samples
#    (family IDs are the tool names)
#  * name of output file

# Output:
# -------
#  * tab-separated file with schema
#    (father1
#    mother1
#    child1
#    father2
#    mother2
#    child2
#    count)
#    containing the genotype combinations and their counts.
#    (Phasing information and multiallelic sites are ignored.)


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


import argparse

gt_map = {
    '..': '00',
    '.0': '00',
    '0.': '00',
    '00': '00',
    '.1': '01',
    '1.': '01',
    '01': '01',
    '10': '01',
    '11': '11'
}

def get_families(ped_path, tools):
    expected_ped_header = \
    '\t'.join(['#Family ID', 'Individual ID', 'Paternal ID', 'Maternal ID', 'Gender', 'Phenotype']) + '\n'
    families = [None, None]
    with open(ped_path, 'r') as ped:
        assert ped.readline() == expected_ped_header, 'Header must be "' + expected_ped_header + '"'
        for line in ped:
            record = line[:-1].split('\t')
            if record[2] == '0':
                continue
            assert record[0] in tools, '"' + record[0] + '" is not in tools'
            family_idx = tools.index(record[0])
            family = [record[2], record[3], record[1]]
            families[family_idx] = family
    return families

def calculate_confusion_matrix(txt_path, families):
    new_header_order = families[0] + families[1] + ['count']
    confusion_counts = {}
    with open(txt_path, 'r') as txt:
        header = txt.readline()[:-1].split('\t')
        header_idx_map = {}
        for i, item in enumerate(header):
            header_idx_map[i] = new_header_order.index(item)
        for line in txt:
            record = line[:-1].split('\t')

            new_gt_list = [None] * 6
            valid_bialelleic_combination = True
            for i, gt in enumerate(record[:-1]):
                if gt not in gt_map:
                    valid_bialelleic_combination = False
                    break
                new_gt_list[header_idx_map[i]] = gt_map[gt]
            if not valid_bialelleic_combination:
                continue

            new_gt_tuple = tuple(new_gt_list)
            if new_gt_tuple not in confusion_counts:
                confusion_counts[new_gt_tuple] = 0
            confusion_counts[new_gt_tuple] += int(record[-1])

    return confusion_counts

def write_confusion_matrix(out_path, confusion_matrix):
    new_header_list = ['father1', 'mother1', 'child1', 
                       'father2', 'mother2', 'child2', 
                       'count']
    with open(out_path, 'w') as out:
        out.write('\t'.join(new_header_list) + '\n')
        sorted_keys = sorted(confusion_matrix.keys())
        for key in sorted_keys:
            record = list(key) + [str(confusion_matrix[key])]
            out.write('\t'.join(record) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('tool1')
parser.add_argument('tool2')
parser.add_argument('aggregate_counts_file')
parser.add_argument('pedigree_file')
parser.add_argument('output_file')
args = parser.parse_args()

ped_path = args.pedigree_file
txt_path = args.aggregate_counts_file
tools = [args.tool1, args.tool2]
out_path = args.output_file

families = get_families(ped_path, tools)
confusion_counts = calculate_confusion_matrix(txt_path, families)
write_confusion_matrix(out_path, confusion_counts)
