# Implementation Plan 2026-04-25

## Scope

Implement the 7 TODO items from `README.md` in the current Python package and keep the Streamlit entry flow working.

## Current State

- Agents currently have:
  - `team_id`
  - `mental_values`
  - one global `similarity_threshold`
- Houses currently have:
  - `FeatureType`
  - `position`
  - optional `agent`
- The move rule currently sends an unhappy agent to any random empty house.
- Tests cover the current simplified agent, city, app, and utility behavior.

## Implementation Order

1. Add explicit configuration objects/constants for:
   - per-group similarity threshold distributions
   - pairwise cultural distance multipliers
   - per-group income distributions
   - housing affordability ratio
   - location multiplier range
2. Extend domain models:
   - add income and per-agent threshold to `Agent`
   - add `location_multiplier` and `property_value` to `Feature`
3. Extend city initialization:
   - sample agent threshold from the group distribution
   - sample income from the group distribution
   - assign fixed location multipliers to houses
   - compute property values from neighborhood income and location multiplier
4. Extend similarity calculation:
   - incorporate pairwise cultural distance multiplier
   - incorporate income similarity as an additional attribute
5. Extend happiness logic:
   - unhappy if social similarity is below the agent threshold
   - unhappy if current house cost exceeds affordability limit
6. Restrict movement:
   - move only to empty houses affordable to the agent
   - if no affordable empty house exists, keep the agent in place for that step
7. Update CLI/Streamlit wiring:
   - thread new parameters through internal construction
   - keep defaults so the app still runs without extra user input
8. Add and update unit tests for every behavior above.

## Task Breakdown By TODO

### 1. Similarity threshold distribution per group

Implementation:
- Replace one global `similarity_threshold` with per-agent sampled threshold values.
- Pass a mapping such as `{team_id: {"mean": x, "std": y}}` during city instantiation.
- Store the sampled threshold on each agent.

Tests:
- verify thresholds are sampled from the configured group parameters
- verify groups can receive different thresholds
- verify existing happiness logic uses the sampled threshold

### 2. Pairwise cultural difference per group pair

Implementation:
- Add a symmetric lookup for group-pair multipliers, for example `(1, 2) -> 1.2`.
- Multiply the baseline cultural distance between two agents by the configured pair factor before converting to similarity.

Tests:
- verify the pair multiplier changes similarity between two otherwise identical comparisons
- verify lookup symmetry for `(a, b)` and `(b, a)`

### 3. Income distribution per group

Implementation:
- Add `income` to `Agent`.
- Sample income from a per-group distribution during city instantiation.
- Include income in similarity evaluation as an additional component.

Tests:
- verify agents receive income from the correct group distribution
- verify income affects similarity/happiness as expected

### 4. Property value from neighborhood income

Implementation:
- Add `property_value` to `Feature`.
- For each house, compute:
  - `base_value = 0.5 * 20 * mean(existing_neighbor_incomes)`
  - then apply the location multiplier if confirmed by operator
- Recompute property values after moves if the value depends on current neighbors.

Tests:
- verify property value uses only existing immediate neighbors
- verify empty or edge neighborhoods are handled deterministically

### 5. Fixed location multiplier per house

Implementation:
- Add `location_multiplier` to `Feature`.
- Assign a fixed multiplier in `[0.5, 1.5]` when the city is instantiated.
- Keep it stable across simulation steps.

Tests:
- verify each house receives a multiplier in range
- verify multiplier remains unchanged after agent movement

### 6. Income satisfaction

Implementation:
- Define affordability limit:
  - `0.6 * 20 * agent.income`
- Mark an agent unhappy if:
  - social similarity is below threshold, or
  - current house value is above affordability limit

Tests:
- verify an otherwise socially satisfied agent becomes unhappy when house cost is too high
- verify a socially unhappy agent is still unhappy even when the house is affordable

### 7. Affordable moves only

Implementation:
- Replace random empty-house selection with affordable empty-house selection.
- Choose randomly among affordable empty houses.
- If none are affordable, do not move the agent.

Tests:
- verify agents only move into affordable empty houses
- verify agents stay put when no affordable destination exists

## Proposed Code Changes

- `schelling_mod/agent.py`
  - add income support
  - add threshold distribution sampling support
  - update happiness logic
- `schelling_mod/feature.py`
  - add `property_value`
  - add `location_multiplier`
- `schelling_mod/city.py`
  - instantiate richer house and agent state
  - add property-value calculation
  - add affordable-empty-house lookup
- `schelling_mod/utils.py`
  - add helper functions for pairwise multiplier lookup and income/social similarity composition
- `schelling_mod/app.py`
  - define default configuration for new model parameters
  - pass configuration into `Schelling` and `City`
- `tests/test_agent.py`
  - extend agent threshold, income, and happiness tests
- `tests/test_city.py`
  - add initialization, property value, and movement restriction tests
- `tests/test_utils.py`
  - add pairwise multiplier and similarity composition tests
- `tests/test_app.py`
  - update parser and defaults if new parameters become user-configurable

## Unclear Points For Operator

1. For TODO 1, should the threshold distribution be normal, uniform, or configurable by distribution type?
2. For TODO 1, do you want the current CLI `--similarity_threshold` removed, or kept as a fallback default for all groups?
3. For TODO 2, should the cultural difference multiplier scale distance directly, or scale similarity after distance is computed?
4. For TODO 3, what distribution should income use per group: normal, log-normal, or fixed mean/std normal?
5. For TODO 3, should income ever be allowed to become non-positive after sampling?
6. For TODO 4, does "immediate neighbors" mean Moore radius 1 including diagonals, or only von Neumann neighbors?
7. For TODO 4, when a house has no existing neighbors, what should its property value be?
8. For TODO 4 and 5, is final house value:
   - `location_multiplier * (0.5 * 20 years income of neighbors)`, or
   - two separate concepts where multiplier does not affect affordability?
9. For TODO 6 and 7, should property values be recomputed after each move within the same iteration, or once per full simulation step?
10. For TODO 7, if multiple affordable empty houses exist, is random choice acceptable?

## Execution Notes

- Implementation should wait for operator answers to the unclear points above before changing simulation semantics.
- Non-ambiguous preparatory refactors can be done earlier if needed, but the movement/economy rules should not be finalized without those answers.
