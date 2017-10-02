#!/usr/bin/env python
"""
This script generates a pedigree file
from the sample names of the input vcfs.
Out of the 6 VCF files, two trios will be 
recorded in the pedigree file, one for each tool.


Arguments:
    tool1: (string) name of tool1
    tool2: (string) name of tool2
    father1, mother1, child1:
        paths to corresponding vcf files
        generated by tool1
    father2, mother2, child2:
        paths to corresponding vcf files
        generated by tool2

Output:
    Pedigree file with two families
    tool1 and tool2 are used as familiy names.
    Relationships are directly assumed 
    from the order of the inputs
    (i.e. f1, m1, c1, f2, m2, c2).
"""

import pysam
import argparse

def get_family_samples(family):
    family_samples = []
    for member in family:
        with pysam.VariantFile(member, 'r') as vcf:
            sample = str(vcf.header.samples[0])
            family_samples.append(sample)
    return family_samples

parser = argparse.ArgumentParser()
parser.add_argument('tool1')
parser.add_argument('tool2')
parser.add_argument('father1')
parser.add_argument('mother1')
parser.add_argument('child1')
parser.add_argument('father2')
parser.add_argument('mother2')
parser.add_argument('child2')
parser.add_argument('output_ped')
args = parser.parse_args()

family1 = [f for f in [args.father1, args.mother1, args.child1]]
family2 = [f for f in [args.father2, args.mother2, args.child2]]
family1_samples = get_family_samples(family1)
family2_samples = get_family_samples(family2)
families = [family1_samples, family2_samples]
tools = [args.tool1, args.tool2]

with open(args.output_ped, 'w') as ped:
    header_list = ['#Family ID', 'Individual ID', 'Paternal ID', 'Maternal ID', 'Gender', 'Phenotype']
    ped.write('\t'.join(header_list) + '\n')
    
    for tool, family in zip(tools, families):
        father_record = [tool, family[0], '0', '0', '1', '-9']
        mother_record = [tool, family[1], '0', '0', '2', '-9']
        child_record = [tool, family[2], family[0], family[1], '0', '-9']
        ped.write('\n'.join(['\t'.join(r) for r in [father_record, mother_record, child_record]]) + '\n')
