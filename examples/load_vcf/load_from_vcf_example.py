#!/usr/bin/env python

# This is an example script showing how to load data
# from a merged VCF(.GZ) file (containing all 2x3 samples)
# and a common PED file (describinig the identity of each sample)
# (Family ID in the PED file must match exactly with
#  the user-supplied tool names)

import sys
import numpy
sys.path.append('../../geck')
from data import GeckData


tool_names = ['bwa+gatk', 'gral+hc']
vcf_file = './6sample_example_chr22.vcf.gz'
ped_file = './example.ped'

for var_type in ['snp', 'indel', 'del', 'ins', 'mnp', 'all']:
    data = GeckData()
    data.load_biallelic_variants_from_merged_vcf_and_pedigree(
        var_type,
        tool_names,
        vcf_file,
        ped_file)
    print var_type + ': ' + str(sum(data.data_dict.values()))
    # tab-delimited csv with header and one entry each line
    data.save_data('./' + var_type +'_counts_in_list_format.txt')
