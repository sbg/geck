#!/usr/bin/env python

# This script processes a pair of fasq files, 
# containing the two reads of each pair 
# of paired-end Illumina reads.

# Both files need to be sorted by readname.

# The script sequentially processes the two files
# simultaneously, and weeds out
#   - Duplicates: when the same file contains the same read multiple times
#     (because they are sorted by readname, these are always consecutive.)
#   - Unpaired reads: when one file contains a read that does not
#     have a matching read in the other file.

# This is called "purging".
# The output are the purged versions of the input files.


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

"""Usage:
    run_fastq_purge [--debug] <fq1_in> <fq2_in> [(-o <fq1_out> <fq2_out>)]
"""
from __future__ import unicode_literals
from docopt import docopt
from fastq_purge import purge_fastq_pair, check_consistency

if __name__ == '__main__':
    args = docopt(__doc__)

    debug = args['--debug']
    fq1_in = args['<fq1_in>']
    fq2_in = args['<fq2_in>']
    if args['-o']:        
        fq1_out = args['<fq1_out>']
        fq2_out = args['<fq2_out>']
    else:
        fq1_out = '.'.join(fq1_in.split('.')[:-1]) + '_purged.' + fq1_in.split('.')[-1]
        fq2_out = '.'.join(fq2_in.split('.')[:-1]) + '_purged.' + fq2_in.split('.')[-1]

    if debug:
        print('Checking consistency of Fastq files ' + fq1_in + ' and ' + fq2_in)
        if check_consistency(fq1_in, fq2_in):
            print('Consistent.')
            print('Exiting.')
            exit(0)

    print('Purging Fastq files, ' + fq1_in + ' and ' + fq2_in)
    purge_fastq_pair(fq1_in, fq2_in, fq1_out, fq2_out)
    print('Finished.')

    if debug:
        print('Checking consistency of Fastq files, ' + fq1_out + ' and ' + fq2_out)
        if check_consistency(fq1_out, fq2_out):
            print('Consistent.')
