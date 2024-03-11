from setuptools import setup, find_packages

setup(
    name='hello_pt',
    version='0.1.0',
    packages=find_packages(include=['hello_pt', 'hello_pt.*'])
)
