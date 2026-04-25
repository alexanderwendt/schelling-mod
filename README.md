# schelling-mod

Modified version of Schelling's segregation simulator that includes economic factors.

## Credits

This project is based on the Streamlit Schelling simulator from
https://github.com/adilmoujahid/streamlit-schelling.
Thanks to Adil Moujahid. Blog: https://adilmoujahid.com/.

## Python Version

This project is now configured for Python `3.14`.

## Dependencies

The runtime dependencies are:

- `numpy==2.4.4`
- `matplotlib==3.10.9`
- `streamlit==1.56.0`

These versions were selected from the current PyPI releases that support Python `3.14`.

## Setup

### Option 1: Conda recommended

This repository already assumes a conda environment named `schelling314`.

Activate the environment:

```powershell
conda activate schelling314
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you want conda to recreate the environment definition, use:

```powershell
conda env update -f environment.yml --prune
```

### Option 2: Pip only

If you do not want to use conda:

```powershell
python -m pip install -r requirements.txt
```

## How To Run

The main entry point is `main.py`.

### Run the Streamlit app

```powershell
conda activate schelling314
streamlit run main.py
```

This starts the interactive UI in your browser.

### Run the simulation from the command line

```powershell
conda activate schelling314
python main.py --run_simulation
```

Optional arguments:

- `--population_size`
- `--empty_ratio`
- `--similarity_threshold`
- `--iterations`

Example:

```powershell
conda activate schelling314
python main.py --run_simulation --population_size 2500 --empty_ratio 0.2 --similarity_threshold 0.4 --iterations 10
```

## Requirements File Or YAML

If you are using conda, `environment.yml` is the better primary environment file because it can pin the Python version
and describe the whole environment in one place. `requirements.txt` is still useful for pip-based installs and for the
`pip` section inside `environment.yml`.

Practical recommendation for this repository:

- Keep `environment.yml` as the primary setup file.
- Keep a small `requirements.txt` with direct Python dependencies only.

## Notes

The `schelling314` conda environment on this machine is Python `3.14.4`, but the project packages were not installed in
that environment at the time of this update, so the application was not fully executed during verification.

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
