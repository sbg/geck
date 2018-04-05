#!/usr/bin/env python


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

class Read:
    def __init__(self):
        self.name = ''
        self.sequence = ''
        self.qualities = ''
    
    def print_fq(self):
        return '\n'.join([
                '@' + self.name,
                self.sequence,
                '+',
                self.qualities
            ]) + '\n'
    
    def _name_to_key(self):
        li = self.name.split(':')
        key = []
        for elem in li:
            try:
                val = int(elem)
            except:
                val = elem
            key.append(val)
        return tuple(key)
    
    def __lt__(self, other):
        return self._name_to_key() < other._name_to_key()
    
    def __le__(self, other):
        return self._name_to_key() <= other._name_to_key()
    
    def __gt__(self, other):
        return self._name_to_key() > other._name_to_key()
    
    def __ge__(self, other):
        return self._name_to_key() >= other._name_to_key()


def fastq_reads(fq_stream):
    while True:
        r = Read()
        name = fq_stream.readline()
        if len(name) == 0:
            raise StopIteration
        assert name[0] == '@', ('read name line should start with "@". It is ' + name)
        r.name = name[1:-1]
        r.sequence = fq_stream.readline()[:-1]
        assert fq_stream.readline() == '+\n'
        r.qualities = fq_stream.readline()[:-1]
        yield r


def unique_fastq_reads(fq_stream):
    prev_name = ''
    for r in fastq_reads(fq_stream):
        if r.name != prev_name:
            prev_name = r.name
            yield r


def check_sorting(fq_path):
    with open(fq_path, 'r') as fq:
        reads = fastq_reads(fq)
        r1 = reads.next()
        r2 = reads.next()
        while True:
            if r1 > r2:
                print(r1.print_fq())
                print(r2.print_fq())
                return False
            try:
                r1 = r2
                r2 = reads.next()
            except StopIteration:
                return True


def check_consistency(fq1_path, fq2_path):
    with open(fq1_path, 'r') as fq1, open(fq2_path, 'r') as fq2:
        reads1 = fastq_reads(fq1)
        reads2 = fastq_reads(fq2)
        for r1 in reads1:
            try:
                r2 = reads2.next()
            except StopIteration:
                print('Extra line in fq_1:\n' + r1.print_fq())
                return False
            if r1.name != r2.name:
                print('Non-matching lines:\n' + 'fq_1:\n' + r1.print_fq() + '\nfq_2:\n' + r2.print_fq())
                return False
        try:
            r2 = reads2.next()
            print('Extra line in fq_2:\n' + r2.print_fq())
        except StopIteration:
            return True


def purge_fastq_pair(fq1_in_path, fq2_in_path, fq1_out_path, fq2_out_path):
    with open(fq1_in_path, 'r') as fq1_in, \
        open(fq2_in_path, 'r') as fq2_in, \
        open(fq1_out_path, 'w') as fq1_out, \
        open(fq2_out_path, 'w') as fq2_out:
        
        reads1 = unique_fastq_reads(fq1_in)
        reads2 = unique_fastq_reads(fq2_in)
        
        r1 = reads1.next()
        r2 = reads2.next()
        while True:
            try:
                if r1.name == r2.name:
                    fq1_out.write(r1.print_fq())
                    fq2_out.write(r2.print_fq())
                    r1 = reads1.next()
                    r2 = reads2.next()
                else:
                    if r1 < r2:
                        r1 = reads1.next()
                    else:
                        r2 = reads2.next()
            except StopIteration:
                return
