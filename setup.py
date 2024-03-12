"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
import codecs

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


### Set up tools to get version
def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


### Do the setup
setup(
    name='NeuralFoil',
    author='Peter Sharpe',
    version=get_version("neuralfoil/__init__.py"),
    description='NeuralFoil is an airfoil aerodynamics analysis tool using physics-informed machine learning, in pure Python/NumPy.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/peterdsharpe/NeuralFoil',
    author_email='pds@mit.edu',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='python machine learning analysis optimization aerospace airplane cfd aircraft hydrodynamics aerodynamics sailing propeller airfoil xfoil design mdo mdao physics informed neural network',
    packages=["neuralfoil"],  # find_packages(exclude=['docs', 'media', 'examples', 'studies'])
    python_requires='>=3.7',
    install_requires=[
        'numpy >= 1',
        'aerosandbox >= 4.2.3'
    ],
    extras_require={
        "training": [
            'torch',
            'ray',
            'polars',
            'tqdm'
        ],
        "test"    : [
            'pytest',
            'nbval'
        ],
        "docs"    : [
            'sphinx',
            'furo',
            'sphinx-autoapi',
        ],
    },
    include_package_data=True,
    package_data={
        'NN parameters': ['*.npz'],  # include the weights and biases for the neural networks
    },
    project_urls={  # Optional
        'Source'     : 'https://github.com/peterdsharpe/NeuralFoil',
        'Bug Reports': 'https://github.com/peterdsharpe/NeuralFoil/issues',
    },
)
