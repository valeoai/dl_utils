from setuptools import setup
from setuptools import find_packages


setup(name='dl-utils',
      version='0.1',
      description='Valeo\'s library for Deep learning training',
      author='Valeo',
      install_requires=['numpy>=1.9.1', 'matplotlib', 'scikit-learn', 'keras'],
      packages=find_packages())
