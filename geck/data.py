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

import numpy as np
from pysam import VariantFile
import itertools


class GeckData:
    """A container for loading and storing aggregate trio genotype data

    The data about the observed joint histogram is stored in a dictionary,
    where keys represent the genotype trio combinations,
    and values are the non-negative counts.

    :ivar tool_names: specifies the names
          (and most importantly the sort order) of the tools (list)
    :ivar data_dict: dictionary of genotypes -> counts

    """
    def __init__(self):
        self.tool_names = None
        self.data_dict = None

    def load_biallelic_variants_from_merged_vcf_and_pedigree(
            self,
            variant_type,
            tool_names,
            vcf_fullpath,
            ped_fullpath):
        """
        :param variant_type: type of variants to be extracted
            'snp': length(REF) == length(ALT) == 1
            'indel': length(REF) >=1 OR length(ALT) >= 1
            'del': length(REF) >= 1 AND length(ALT) == 1
            'ins': length(REF) == 1 AND length(ALT) >= 1
            'mnp': length(REF) >= 1 AND length(ALT) >= 1
            'all': all (bi-allelic) variants
        :type variant_type: str
        :param tool_names: list of two tool names, they should exactly match
                           the Family IDs in pedigree file
        :param vcf_fullpath: path to vcf or vcf.gz file
        :type vcf_fullpath: str
        :param ped_fullpath: path to pedigree file
        :type ped_fullpath: str
        :return: None (populates data_dict)
        """

        # load sample names from pedigree file
        trios = {}
        with open(ped_fullpath, 'r') as ped:
            for line in ped:
                if line[0] == '#':
                    continue
                record = line[:-1].split('\t')
                assert len(record) >= 5, \
                    'Error: col 1-5 of pedigree file must be: ' + \
                    'Family ID, Individual ID, Paternal ID, \
                     Maternal ID, Gender'
                family_id, \
                    individual_id, \
                    paternal_id, \
                    maternal_id, \
                    gender_id = record[:5]
                if paternal_id == '0' or maternal_id == '0':
                    # skip entry corresponding to the parents ancestry
                    continue
                trio = (paternal_id, maternal_id, individual_id)
                trios[family_id] = trio
        samples_in_ped = [trios[k][i] for
                          k, i in itertools.product(trios.keys(), range(3))]

        # load genotypes from variant file
        vcf_reader = VariantFile(vcf_fullpath)
        assert vcf_reader.is_open, 'Error: opening variant file'
        samples_in_vcf = [vcf_reader.header.samples[i]
                          for i in range(len(vcf_reader.header.samples))]
        for s in samples_in_vcf:
            assert s in samples_in_ped, \
                'Error: sample in variant file is missing \
                 from pedigree file: ' + str(s)

        map_gt = {None: '0',  # 'no call' is considered to be 'reference'
                  0: '0',
                  1: '1'}

        # set up variant type filter
        if variant_type == 'snp':
            def _is_variant_correct_type(var):
                return len(var.ref) == 1 and len(var.alts[0]) == 1
        elif variant_type == 'indel':
            def _is_variant_correct_type(var):
                return len(var.ref) > 1 or len(var.alts[0]) > 1
        elif variant_type == 'del':
            def _is_variant_correct_type(var):
                return len(var.ref) > 1 and len(var.alts[0]) == 1
        elif variant_type == 'ins':
            def _is_variant_correct_type(var):
                return len(var.ref) == 1 and len(var.alts[0]) > 1
        elif variant_type == 'mnp':
            def _is_variant_correct_type(var):
                return len(var.ref) > 1 and len(var.alts[0]) > 1
        else:
            def _is_variant_correct_type(var):
                return True

        data_dict = {}
        for variant in vcf_reader:
            if not _is_variant_correct_type(variant):
                continue
            if len(variant.alts) != 1:  # skip multi-allelic variants
                continue
            trio_gts = []
            for tool in tool_names:
                trio_gt = []
                for sample in trios[tool]:
                    allele_indices = variant.samples[sample].allele_indices
                    sample_gt = ''.join([map_gt[ai] for ai in allele_indices])
                    trio_gt.append(sample_gt)
                trio_gts.append(tuple(trio_gt))
            trio_gts = tuple(trio_gts)
            data_dict[trio_gts] = data_dict[trio_gts] + 1 \
                if trio_gts in data_dict else 1
        self.data_dict = data_dict

    def load_biallelic_snps_from_merged_vcf_and_pedigree(self,
                                                         tool_names,
                                                         vcf_fullpath,
                                                         ped_fullpath):
        """Aggregates genotype counts of bi-allelic SNPs

        (For loading different type of variants see
         GeckData.load_variants_from_merged_vcf_and_pedigree() )

        It uses the pedigree file to identify which sample names in the VCF
        correspond to which tool, and which family member.
        Non bi-allelic SNP entries are skipped.
        Usually produces no ((00,00,00),(00,00,00)) counts, requiring
        setting it manually to a
        high enough value (say 5e7, size of dbSNP)

        :param tool_names: list of two tool names, they should exactly match
                           the Family IDs in pedigree file
        :param vcf_fullpath: path to vcf or vcf.gz file
        :type vcf_fullpath: str
        :param ped_fullpath: path to pedigree file
        :type ped_fullpath: str
        :return: None (populates data_dict)

        """
        self.load_biallelic_variants_from_merged_vcf_and_pedigree(
            'snp',
            tool_names,
            vcf_fullpath,
            ped_fullpath)

    def load_file(self, tool_names, input_fullpath, header=True):
        """Loads data from file


        :param tool_names: names of the two tools
        :type tool_names: list
        :param input_fullpath: location of input file, which
            should
                - be tab-delimited
                - contain the column:
                  father1, mother1, child1, father2, mother2, child2, count
                - have "00", "01", "11" genotypes in columns
                  (father/mother/child)(1/2):
                - have non-negative integers in column "count"
        :type input_fullpath: str
        :param header: True if the first line is a header (default: True)
        :type header: bool
        :return: None (populates data_dict)

        """
        self.tool_names = tool_names
        default_columns = ['father1', 'mother1', 'child1',
                           'father2', 'mother2', 'child2',
                           'count']
        with open(input_fullpath, 'r') as in_file:
            # load (or initialize) header
            if header:
                columns = in_file.readline()[:-1].split('\t')
                assert set(columns) == set(default_columns), \
                    'header should contain the columns ' \
                    + str(default_columns)
                column_index = {}
                idx = 0
                for item in columns:
                    column_index[item] = idx
                    idx += 1
            else:
                column_index = dict([(default_columns[i], i)
                                     for i in range(len(default_columns))])
            # load data
            data_dict = {}
            for line in in_file:
                record = line[:-1].split('\t')
                g1 = (record[column_index['father1']],
                      record[column_index['mother1']],
                      record[column_index['child1']])
                g2 = (record[column_index['father2']],
                      record[column_index['mother2']],
                      record[column_index['child2']])
                assert (g1, g2) not in data_dict, \
                    'input file should not contain duplicate entries \
                     (duplicate entry with ' + str((g1, g2)) + ')'
                data_dict[(g1, g2)] = int(record[column_index['count']])
        self.data_dict = data_dict

    def load_matrix(self, tool_names, joint_histogram, g1g2_labels):
        """Loads data from N[G1,G2] matrix, using the labels in G1G2_labels

        :param tool_names: names of the two tools
        :type tool_names: list
        :param joint_histogram: 27x27 array of integers
        :type joint_histogram: numpy.array
        :param g1g2_labels: list of (G1, G2)
        :type g1g2_labels: list
        :return: None (updates self.data_dict)

        """
        self.tool_names = tool_names
        data_dict = {}
        for G1_idx, G1 in enumerate(g1g2_labels):
            for G2_idx, G2 in enumerate(g1g2_labels):
                data_dict[(G1, G2)] = joint_histogram[G1_idx, G2_idx]
        self.data_dict = data_dict

    def get_data_matrix(self, sort_order):
        """Organizes  counts in data_dict into an array.

        Follows the order of components defined by order.
        Entries of order not in data_dict are filled with zero.
        Entries of data_dict not in order are not present in the output.

        :param sort_order: list of all keys present in data_dict
        :type sort_order: list
        :return: numpy array of the counts (integers)

        """
        data_matrix = np.zeros((len(sort_order), len(sort_order)))
        for k in self.data_dict:
            idx1 = sort_order.index(k[0])
            idx2 = sort_order.index(k[1])
            data_matrix[idx1, idx2] = self.data_dict[k]
        return data_matrix

    def save_data(self, out_fullpath, format='list', sort_order=None):
        """
        Saves the data in the

        :param out_fullpath: path to output file
        :type out_fullpath: str
        :param format: format of the output
            'list':  same format which load_file() accepts
            'matrix': matrix form,
                      order of rows/columns must be specified by sort_order
        :type format: str
        :param sort_order: list of all keys present in data_dict
        :type sort_order: list
        :return: None (writes to file)

        """
        if format == 'list':
            with open(out_fullpath, 'w') as out_file:
                header = '\t'.join(['father1', 'mother1', 'child1',
                                    'father2', 'mother2', 'child2',
                                    'count']) + '\n'
                out_file.write(header)
                sorted_keys = sorted(self.data_dict.keys())
                for key in sorted_keys:
                    g1 = list(key[0])
                    g2 = list(key[1])
                    record = '\t'.join(g1 + g2 +
                                       [str(self.data_dict[key])]) + '\n'
                    out_file.write(record)

        elif format == 'matrix' and sort_order is not None:
            data_matrix = self.get_data_matrix(sort_order).astype(int)
            np.savetxt(out_fullpath, data_matrix, fmt='%i', delimiter='\t')
        else:
            raise ValueError('format must be "list" or "matrix", \
                              and sort_order must be specified')
