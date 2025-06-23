from setuptools import setup
from setuptools import find_packages

setup(name='ood',
        version='0.1',
        description='CardOOD SQL Cardinality estimator',
        license='MIT',
        install_requires=['numpy', 'torch', 'scipy'],
        packages=find_packages())
