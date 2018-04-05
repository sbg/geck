#!/usr/bin/env python

# This is an example script showing how to load data
# from a merged VCF(.GZ) file (containing all 2x3 samples)
# and a common PED file (describinig the identity of each sample)
# (Family ID in the PED file must match exactly with
#  the user-supplied tool names)


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
    print(var_type + ': ' + str(sum(data.data_dict.values())))
    # tab-delimited csv with header and one entry each line
    data.save_data('./' + var_type +'_counts_in_list_format.txt')
