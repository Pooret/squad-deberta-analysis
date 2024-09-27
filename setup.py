from setuptools import setup, find_packages


# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='squad_deberta_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'run_experiment=scripts.run_experiment:main',
        ],
    },
)
