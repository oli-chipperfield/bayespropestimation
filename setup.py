#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Oliver Chipperfield",
    author_email='omc1985@googlemail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Class method for the Bayesian estimation and comparison of proportions',
    install_requires=requirements,
    license="MIT license",
    long_description='test long description',  #readme + '\n\n' + history,
    include_package_data=True,
    keywords='bayespropestimation',
    name='bayespropestimation',
    packages=find_packages(include=['bayespropestimation', 'bayespropestimation.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/oli-chipperfield/bayespropestimation',
    version='0.1.1',
    zip_safe=False,
)
