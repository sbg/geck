import io
from setuptools import setup, find_packages

setup(
    name='geck',
    version = '1.0',
    url = 'https://github.com/sbg/geck.git',
    author='Peter Komar',
    author_email='peter.komar@sbgdinc.com',
    description = 'Genotype Error Comparator Kit -- estimates and compares the accuracies of two genotyping methods using only their joint result on a family trio.',
    packages = find_packages(),    
    install_requires=io.open('requirements.txt').read().splitlines(),
    include_package_data=True,
)
