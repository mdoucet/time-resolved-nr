# Repository for the analysis of time-resolved NR data with reinforcement learning

## Installation

Clone the repo:
```
git clone https://github.com/mdoucet/time-resolved-nr
cd time-resolved-nr
```


Then create the environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ipykernel # Optional: To use the jupyter notebooks
pip install -e .
```

Examples from the paper are in the `notebooks` directory, and examples from users will be added to
the `examples` directory.
