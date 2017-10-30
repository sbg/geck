#!/usr/bin/env python

# This script calculates true benchmarking metrics from raw gt counts, obtained from 2-sample variant file (truth + calls).
# It performs two tasks:

# 1. Sanitizes raw counts, i.e.
#  - removes multiallelic counts
#  - disregards phasing (aggregates over them)
#  - assumes reference allele for missing genotypes (i.e. '.' == '0')

# 2. Calculates benchmarking metrics.

# Input:
# -------
#  * raw_counts_txt_1: TXT file produced by aggregate-calls-with-truth workflow for pipeline 1
#  * raw_counts_txt_2: TXT file produced by aggregate-calls-with-truth workflow for pipeline 2

# Outputs:
# --------
#  * tab-separated confusion matrix with scheme
#    (true genotype,
#    	called genotype,
#    	counts from comparing tool1 calls with truth,
#    	counts from comparing tool2 calls with truth)
#  * benchmarking metrics JSON
#    lvl1: hard/soft
#    lvl2: percision/recall/Fscore
#    lvl3: tool1/tool2/diff


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
import json

gt_map = {
    
    # homozygous reference
    '..': '00',
    '.0': '00',
    '0.': '00',
    '00': '00',
    
    # heterozygous alternate
    '.1': '01',
    '1.': '01',
    '01': '01',
    '10': '01',
    
    # homozygous alternate
    '11': '11'
}

count_map = {
    # (true_gt, false_gt) : category

# False positives
 ('00', '01'): 'FP',
 ('00', '11'): 'FP',

# False negatives
 ('01', '00'): 'FN',
 ('11', '00'): 'FN',

# "Mixed-up" positives
 ('01', '11'): 'MP',
 ('11', '01'): 'MP',
    
# True positives
 ('01', '01'): 'TP',
 ('11', '11'): 'TP'
}

def calculate_confusion_matrix(gt_counts_txt):
    confusion_matrix = {}  # (true_gt, called_gt) : count
    with open(gt_counts_txt, 'r') as txt:
        header_record = txt.readline()[:-1].split('\t')
        expected_header_record = ['truth', 'calls', 'count']
        assert set(header_record) == set(expected_header_record)
        for line in txt:
            record = line[:-1].split('\t')
            record_dict = {}
            for idx, item in enumerate(record):
                column_name = header_record[idx]
                record_dict[column_name] = item
            raw_key = (record_dict['truth'], record_dict['calls'])
            key = []
            valid_gt = True
            for gt in raw_key:
                if gt not in gt_map:
                    valid_gt = False
                    break
                key.append(gt_map[gt])
            key = tuple(key)
            if not valid_gt: continue
            count = int(record_dict['count'])
            if key not in confusion_matrix: confusion_matrix[key] = 0
            confusion_matrix[key] += count
    return confusion_matrix

def calculate_counts(confusion_matrix):
    counts = dict(zip(['FP', 'FN', 'TP', 'MP'], [0]*4))
    for gt in confusion_matrix:
        if gt not in count_map: continue
        key = count_map[gt]
        if key not in counts: counts[key] = 0
        counts[key] += confusion_matrix[gt]
    return counts

def _precision(counts, mode):
    if mode == 'hard':
        return counts['TP'] / float(counts['TP'] + counts['FP'] + counts['MP'])
    if mode == 'soft':
        return (counts['TP'] + counts['MP']) / float(counts['TP'] + counts['FP'] + counts['MP'])

def _recall(counts, mode):
    if mode == 'hard':
        return counts['TP'] / float(counts['TP'] + counts['FN'] + counts['MP'])
    if mode == 'soft':
        return (counts['TP'] + counts['MP']) / float(counts['TP'] + counts['FN'] + counts['MP'])

def _Fscore(counts, mode):
    return 2.0 / (1.0 / _recall(counts, mode) + 1.0 / _precision(counts, mode))

metric_functions = {
    'precision': _precision,
    'recall': _recall,
    'Fscore': _Fscore
}

def calculate_metrics(counts1, counts2):
    counts = [counts1, counts2]
    modes = ['hard', 'soft']
    metrics = ['precision', 'recall', 'Fscore']
    tools = ['tool1', 'tool2', 'diff']
    results_dict = {}
    for mode in modes:
        if mode not in results_dict: results_dict[mode] = {} 
        for metric in metrics:
            if metric not in results_dict[mode]: results_dict[mode][metric] = {} 
            for count, tool in zip(counts, ['tool1', 'tool2']):
                results_dict[mode][metric][tool] = metric_functions[metric](count, mode)
            results_dict[mode][metric]['diff'] = results_dict[mode][metric]['tool2'] - results_dict[mode][metric]['tool1']
    return results_dict

def write_metrics(results, json_path):
    with open(json_path, 'w') as out:
        out.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

def write_confusion_counts(confusion_matrix1, confusion_matrix2, output_path):
    header_list = ['truth', 'calls', 'count_tool1', 'count_tool2']
    with open(output_path, 'w') as out:
        out.write('\t'.join(header_list) + '\n')
        sorted_keys = sorted(confusion_matrix1.keys())
        for key in sorted_keys:
            record = [key[0], key[1], str(confusion_matrix1[key]), str(confusion_matrix2[key])]
            out.write('\t'.join(record) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('raw_counts_txt_1')
parser.add_argument('raw_counts_txt_2')
parser.add_argument('output_txt_name')
parser.add_argument('output_json_name')
args = parser.parse_args()

input_path1 = args.raw_counts_txt_1
input_path2 = args.raw_counts_txt_2
txt_path  = args.output_txt_name
json_path  = args.output_json_name

confusion_matrix1 = calculate_confusion_matrix(input_path1)
confusion_matrix2 = calculate_confusion_matrix(input_path2)
counts1 = calculate_counts(confusion_matrix1)
counts2 = calculate_counts(confusion_matrix2)
write_confusion_counts(confusion_matrix1, confusion_matrix2, txt_path)

results_dict = calculate_metrics(counts1, counts2)
write_metrics(results_dict, json_path)
