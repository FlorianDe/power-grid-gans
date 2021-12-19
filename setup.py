from setuptools import setup, find_packages

setup(
    name='power_grid_gans',
    version='0.0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url='',
    license='',
    author='floriande',
    author_email='',
    description='Power Grid GANs'
)
