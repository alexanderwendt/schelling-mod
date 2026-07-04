# schelling-mod

Modified Schelling segregation simulator with a Streamlit frontend, configurable group dynamics, movement rules, and
economic housing constraints.

![System Overview](doc/images/Overview.png)

## Overview

This project extends a Schelling-style segregation model with:

- configurable 2-5 groups with editable names, colors, population shares, similarity thresholds, and incomes
- default LOTR-inspired group names: `Knights`, `Elves`, `Orcs`, `Dwarves`, `Hobbits`
- pairwise cultural distance between groups, shown in the UI as `<Group A> - <Group B> cultural distance`
- Moore and von Neumann neighborhoods
- cross-shaped street/barrier cells that partition the map into four blocks
- multiple movement modes: `random_empty`, `first_acceptable`, `best_sampled`, `best_available`, `limited_distance`
- movement probabilities for dissatisfied and satisfied agents
- movement sample size, optional movement search radius, and optional satisfied-destination requirement
- seeded random number generation for reproducible maps, sampled attributes, and movement choices
- property values, affordability constraints, and optional density preference
- Streamlit exports for per-run `metrics.csv` and `config.json`

## Model Features

### Groups

Runs support 2-5 active groups. Each group has:

- `name`
- `color`
- `population_share`
- `threshold_mean`
- `threshold_std`
- `income_mean`
- `income_std`

Population shares are normalized internally, so they do not need to sum exactly to `1.0` or `100%`.

### Cultural distance

Every active pair of groups has one cultural-distance value. The UI labels these controls as:

```text
<Group A> - <Group B> cultural distance
```

The current cultural-similarity formula is:

```text
cultural_similarity = max(0, 1 - cultural_distance)
```

Same-group distance is treated as `0`. Distance `0.0` means full similarity, distance `1.0` means zero similarity, and
values above `1.0` remain clamped at zero similarity.

### Neighborhoods and streets

Neighborhood radius is configurable from `1` to `5` in Streamlit.

- `moore`: Chebyshev distance; square neighborhood around the cell.
- `von_neumann`: Manhattan distance; diamond neighborhood around the cell.

Generated maps receive a cross-shaped street/barrier through the center row and center column. Streets are displayed in
black, cannot be occupied, are excluded from neighborhood capacity and property-value inputs, and count as barriers. In
the current implementation they do not block line-of-sight or movement across streets.

### Movement

The movement mode controls how empty destinations are selected:

- `random_empty`: choose a random valid empty house.
- `first_acceptable`: scan randomized candidates and pick the first destination meeting the agent threshold.
- `best_sampled`: sample up to `sample_size` empty houses and choose the best by satisfaction score.
- `best_available`: evaluate all empty houses and choose the best; blocked in Streamlit/CLI when too many empty cells
  would make exhaustive search expensive.
- `limited_distance`: like `best_sampled`, but candidates are constrained to a Manhattan search radius. If no radius is
  configured, the neighborhood radius is used.

Movement parameters:

- `sample_size`: maximum sampled destinations for sampled modes.
- `movement_search_radius`: optional Manhattan radius for destination search.
- `dissatisfied_move_probability`: probability that an unhappy agent attempts movement.
- `satisfied_move_probability`: probability that a happy agent makes an exploratory move.
- `require_satisfied_destination`: reject destinations whose score is below the moving agent's threshold.

When affordability is enabled, destination candidates must also be affordable.

### Random seed reproducibility

The `seed` value initializes NumPy's random generator. With the same config and seed, map generation, threshold and
income samples, random candidate sampling, and tie-breaking movement choices are reproducible.

### Density preference

When density preference is enabled, satisfaction combines cultural similarity with local occupied-neighbor density:

```text
satisfaction_score = (0.9 * social_similarity) + (0.1 * density_ratio)
density_ratio = occupied_neighbor_count / possible_non_street_neighbor_count
```

When disabled, satisfaction uses social similarity only.

## Economic Model

The model separates two price mechanisms.

First, each house has a location multiplier. This follows the intuition of the Alonso-Muth-Mills urban model: central
locations are more valuable because they are closer to the city center. In this simulation, the multiplier is highest in
the map center and declines with distance toward the edges.

Second, each house has an endogenous neighborhood-income component. This follows the gentrification mechanism described
by Guerrieri, Hartley, and Hurst: when higher-income residents move into or near a neighborhood, their presence can bid
up nearby property prices and create displacement pressure for lower-income residents.

The implemented property-value formula uses a default capitalized value:

```text
Dval = MultiplyYear * IncomeShare
Dval = 20 * 0.5 = 10
```

Each house also has an intrinsic location multiplier `ival` in the range `[0.5, 1.5]`. In capitalized value units this
corresponds to `[5, 15]`, because `Dval * ival` gives the location fallback value. Central cells have higher `ival`;
edge cells have lower `ival`.

For property prices, each non-street immediate neighbor slot contributes one value:

```text
occupied neighbor slot = neighbor income
empty neighbor slot = ival
property_value = Dval * mean(neighbor income or ival)
```

Streets are excluded. If there are no usable neighboring house slots, the house uses its own `ival` as the mean value.

Affordability is checked against the final `property_value`:

```text
afford_value = Dval * income
affordable if afford_value >= property_value
```

For an agent with `income = 1.0`, `afford_value = 10`. If affordability constraints are enabled, an agent that cannot
afford its current house is unhappy and may only move to an affordable empty house.

Examples:

- `ival = 1.5`, all neighbors occupied with income `1.0`: `mean = 1.0`, `property_value = 10`
- `ival = 1.5`, 50% neighbors occupied with income `1.0`: `mean = 1.25`, `property_value = 12.5`
- `ival = 0.5`, all neighbors occupied with income `1.0`: `mean = 1.0`, `property_value = 10`
- `ival = 0.5`, 50% neighbors occupied with income `1.0`: `mean = 0.75`, `property_value = 7.5`

## Streamlit Frontend

Run the app:

```powershell
conda activate schelling311
streamlit run main.py
```

Sidebar sections:

- `Population / groups`: population size, empty-house ratio, and group definitions.
- `Group definitions`: group count, name, color, population share, threshold mean/std dev, and income mean/std dev.
- `Preferences / cultural distance`: pairwise cultural distance sliders.
- `Run settings`: iteration count and random seed.
- `Neighborhoods`: radius and `moore` / `von_neumann` type.
- `Movement`: mode, sample size, search radius, movement probabilities, and satisfied-destination requirement.
- `Economics / attractiveness`: affordability constraints and density preference.
- `Export`: download controls after a run.

Main page sections:

- group map with empty houses and street/barrier legend
- property-value heatmap
- mean-similarity chart
- selected-cell inspector for row/column details
- current configuration JSON expander
- summary metrics table
- export buttons for `metrics.csv` and `config.json` after a run

The app loads initial values from `schelling_streamlit_config.json` if present and saves the active configuration when a
simulation is run.

## CLI Usage

Run the command-line simulation:

```powershell
conda activate schelling311
python main.py --run_simulation
```

Load a versioned config JSON:

```powershell
python main.py --run_simulation --config-json path/to/config.json
```

Current CLI arguments:

- `--config-json`
- `--population_size`, `--population-size`
- `--empty_ratio`, `--empty-ratio`
- `--threshold_std_dev`
- `--iterations`
- `--seed`
- `--neighborhood-radius`
- `--neighborhood-type {moore,von_neumann}`
- `--movement-mode {random_empty,first_acceptable,best_sampled,best_available,limited_distance}`
- `--sample-size`
- `--movement-search-radius`
- `--dissatisfied-move-probability`
- `--satisfied-move-probability`
- `--affordability` / `--no-affordability`
- `--density-preference` / `--no-density-preference`
- `--output-dir`

When `--output-dir` is set, CLI runs write `metrics.csv` and `config.json` to that directory. The simulation stops early
in both CLI and Streamlit mode if all agents are satisfied before the configured iteration limit is reached.

Example:

```powershell
python main.py --run_simulation --population-size 2500 --empty-ratio 0.2 --iterations 10 --seed 42 \
  --neighborhood-type von_neumann --movement-mode limited_distance --movement-search-radius 4 --output-dir runs/example
```

## Metrics and Exports

Each metric row includes run id, iteration, satisfaction percentage, mean similarity, mean satisfaction score, move
count, convergence iteration, affordability failures, displacement distance metrics, segregation indexes, and
attractiveness exposure metrics.

Exports:

- Streamlit: download `metrics.csv` and `config.json` after running.
- CLI: write `metrics.csv` and `config.json` when `--output-dir` is provided.

## Python Version

This project targets Python `3.11`.

## Project Structure

```text
schelling-mod/
├── doc/
│   └── images/
│       └── Overview.png
├── schelling_mod/
│   ├── __init__.py
│   ├── agent.py
│   ├── app.py
│   ├── city.py
│   ├── config.py
│   ├── export.py
│   ├── feature.py
│   ├── metrics.py
│   ├── movement.py
│   └── utils.py
├── tests/
├── environment.yml
├── main.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

- `schelling_mod/agent.py`: agent satisfaction, similarity, and affordability.
- `schelling_mod/city.py`: grid generation, street layout, neighborhoods, movement candidate selection, property values.
- `schelling_mod/config.py`: versioned config defaults, migration, groups, movement modes, neighborhood types.
- `schelling_mod/movement.py`: movement mode constants and movement stats.
- `schelling_mod/metrics.py`: per-iteration metric rows and indexes.
- `schelling_mod/export.py`: `metrics.csv` and `config.json` export helpers.
- `schelling_mod/app.py`: Streamlit UI and CLI orchestration.
- `main.py`: compatibility entry point for Streamlit and CLI use.

## Installation

### Option 1: Conda recommended

The repository is configured for a conda environment named `schelling311`.

```powershell
conda env update -f environment.yml --prune
conda activate schelling311
python -m pip install -e .
```

### Option 2: Pip only

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Requirements File Or YAML

If you are using conda, `environment.yml` should be treated as the primary environment file because it defines the
Python version and the installation path in one place. Keep `requirements.txt` for direct pip dependencies.

## Literature

- Guerrieri, Hartley, and Hurst, "Endogenous Gentrification and Housing Price Dynamics":
  https://www.nber.org/papers/w16237
- Journal version, "Endogenous gentrification and housing price dynamics":
  https://www.sciencedirect.com/science/article/pii/S0047272713000297
- Alonso-Muth-Mills model overview:
  https://www.rba.gov.au/publications/rdp/2011/2011-03/alonso-muth-mills-model.html
- Additional reading:
  https://www.sciencedirect.com/science/article/pii/S0264275124000520

## Credits

This project is based on the Streamlit Schelling simulator from
https://github.com/adilmoujahid/streamlit-schelling.

Thanks to Adil Moujahid. Blog: https://adilmoujahid.com/.
