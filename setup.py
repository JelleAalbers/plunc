#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

test_requirements = requirements  # + ['flake8', 'tox', 'coverage','bumpversion']

setup(name='plunc',
      version='0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0',
      description='Poisson Limits Using Neyman Constructions',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/plunc',
      packages=['plunc', 'plunc.statistics', 'plunc.intervals',],
      package_dir={'plunc': 'plunc'},
      install_requires=requirements,
      license="MIT",
      zip_safe=False,
      keywords='pax',
      classifiers=[
          'Development Status :: -1 - Does not work',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
      test_suite='tests',
      tests_require=test_requirements)
