# schelling-mod

Modified version of Schelling's segregation simulator that includes economic factors.

## Credits

This project is based on the Streamlit Schelling simulator from
https://github.com/adilmoujahid/streamlit-schelling.
Thanks to Adil Moujahid. Blog: https://adilmoujahid.com/.

## Python Version

This project targets Python `3.11`.

## Project Structure

The project follows a standard Python package layout:

```text
schelling-mod/
├── schelling_mod/
│   ├── __init__.py
│   ├── agent.py
│   ├── app.py
│   ├── city.py
│   ├── feature.py
│   └── utils.py
├── environment.yml
├── main.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

- `schelling_mod/` contains the application and domain logic.
- `main.py` is a thin entry-point wrapper for Streamlit and CLI use.
- `pyproject.toml` defines the project metadata and Python requirement.

## Dependencies

The runtime dependencies are:

- `numpy==2.4.4`
- `matplotlib==3.10.9`
- `streamlit==1.56.0`

Install the dependencies from `requirements.txt` or `environment.yml`.

## Setup

### Option 1: Conda recommended

This repository assumes a conda environment named `schelling311`.

Activate the environment:

```powershell
conda activate schelling311
```

Install or update the environment from the conda definition:

```powershell
conda env update -f environment.yml --prune
```

Optional: install the package in editable mode:

```powershell
python -m pip install -e .
```

### Option 2: Pip only

If you do not want to use conda:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install is optional for the current workflow because `main.py` already imports the local package, but it
is the standard approach for Python projects with package metadata.

## How To Run

The main entry point remains `main.py`.

### Run the Streamlit app

```powershell
conda activate schelling311
streamlit run main.py
```

This starts the interactive UI in your browser.

### Run the simulation from the command line

```powershell
conda activate schelling311
python main.py --run_simulation
```

Optional arguments:

- `--population_size`
- `--empty_ratio`
- `--similarity_threshold`
- `--iterations`

Example:

```powershell
conda activate schelling311
python main.py --run_simulation --population_size 2500 --empty_ratio 0.2 --similarity_threshold 0.4 --iterations 10
```

### Run as a module

You can also run the packaged app directly:

```powershell
python -m pip install -r requirements.txt
python -m schelling_mod.app --run_simulation
```

## Requirements File Or YAML

If you are using conda, `environment.yml` is the better primary environment file because it can pin the Python version
and describe the whole environment in one place. `requirements.txt` is still useful for pip-based installs and for the
`pip` section inside `environment.yml`.

Practical recommendation for this repository:

- Keep `environment.yml` as the primary setup file.
- Keep a small `requirements.txt` with direct Python dependencies only.

## Validation

The package structure and entrypoints were checked after the refactor. In this workspace, a full runtime launch could
not be completed because `matplotlib` is not installed in the active Python environment.

## TODOs

1. Similarity threshold should be a distribution instead of a single value. Tau should be configurable per group.
2. Cultural difference should vary between group pairs, for example G1-G2: 1.2, G1-G3: 1.5, G1-G4: 2.5.
3. Economy: each group should have an income distribution. This should act as an additional similarity attribute.
4. Property value: the value of a house should be `50% * 20 years income` of the immediate neighbors that exist.
5. Location multiplier: each house should have a fixed base factor from `0.5` to `1.5` depending on location.
6. Income satisfaction: if housing cost is greater than `60% * 20 years income`, the agent becomes unhappy and must move.
7. Moving: unhappy agents may only move into an empty house whose value is less than or equal to `60% * 20 years income`.

## Literature

To read: https://www.sciencedirect.com/science/article/pii/S0264275124000520
