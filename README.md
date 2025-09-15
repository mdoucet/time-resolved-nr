# Analysis of time-resolved NR data with reinforcement learning

[![Python Tests](https://github.com/mdoucet/time-resolved-nr/actions/workflows/python-test.yml/badge.svg)](https://github.com/mdoucet/time-resolved-nr/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/mdoucet/time-resolved-nr/branch/main/graph/badge.svg)](https://codecov.io/gh/mdoucet/time-resolved-nr)

The approach using RL to model time-resolved neutron reflectometry data was first reported
in [M. Doucet et al, J. Phys. Chem. Lett. 2024, 15, 16, 4444–4450](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.4c00467). This repository contained the original version of the demonstration code. 
It has since evolved into a package that can be used to process t-NR data from the Liquids Reflectometer at ORNL.
It can also be extended for other facilities.


## Installation

Clone the repo:
```
git clone https://github.com/mdoucet/time-resolved-nr
cd time-resolved-nr
```

Then create the environment:
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel # Optional: To use the jupyter notebooks
pip install -e .
```

Examples from the paper are in the `notebooks` directory, and examples from users will be added to
the `examples` directory.


## TODO
- Determine scale from first time step
- Option to set scale
- Way to monitor during training
