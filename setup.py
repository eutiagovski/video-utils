# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Video Utils',
    version='0.1.0',
    description='Sample Video Utils',
    long_description=readme,
    author='Tiago Machado',
    author_email='tiagomachadodev@gmail.com',
    url='https://github.com/eutiagovski/python-utils/tree/master/core/video/Capture',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)