# Analysis of time-resolved NR data with reinforcement learning

[![Python Tests](https://github.com/mdoucet/time-resolved-nr/actions/workflows/python-test.yml/badge.svg)](https://github.com/mdoucet/time-resolved-nr/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/mdoucet/time-resolved-nr/branch/main/graph/badge.svg)](https://codecov.io/gh/mdoucet/time-resolved-nr)

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
