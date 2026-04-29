"""Unit tests for application entrypoints."""

import json

import numpy as np

from schelling_mod.app import build_parser
from schelling_mod.app import build_default_streamlit_config
from schelling_mod.app import load_streamlit_config
from schelling_mod.app import MENTAL_VALUES_MAP
from schelling_mod.app import normalize_teams_distribution
from schelling_mod.app import save_streamlit_config
from schelling_mod.app import Schelling


def test_build_parser_reads_expected_cli_values() -> None:
    """The CLI parser should expose the documented simulation options."""
    args = build_parser().parse_args(
        [
            "--run_simulation",
            "--population_size",
            "81",
            "--empty_ratio",
            "0.3",
            "--threshold_std_dev",
            "0.06",
            "--iterations",
            "4",
        ]
    )

    assert args.run_simulation is True
    assert args.population_size == 81
    assert args.empty_ratio == 0.3
    assert args.threshold_std_dev == 0.06
    assert args.iterations == 4


def test_build_parser_known_args_tolerates_streamlit_flags() -> None:
    """Parsing should allow unrelated Streamlit arguments to be ignored."""
    args, unknown = build_parser().parse_known_args(
        [
            "--run_simulation",
            "--server.port",
            "8501",
        ]
    )

    assert args.run_simulation is True
    assert unknown == ["--server.port", "8501"]


def test_streamlit_config_round_trips_to_json_file(tmp_path) -> None:
    """Streamlit settings should persist to and load from a config file."""
    config_path = tmp_path / "schelling_streamlit_config.json"
    config = build_default_streamlit_config()
    config["simulation"]["population_size"] = 1600
    config["teams"]["1"]["population_share"] = 0.2
    config["teams"]["1"]["income_mean"] = 1.5
    config["pairwise_cultural_multipliers"]["1-2"] = 2.0

    save_streamlit_config(config, config_path)

    assert load_streamlit_config(config_path) == config


def test_streamlit_config_load_merges_partial_file_with_defaults(tmp_path) -> None:
    """Missing settings in a saved file should fall back to defaults."""
    config_path = tmp_path / "schelling_streamlit_config.json"
    config_path.write_text(
        json.dumps({"simulation": {"population_size": 4096}}),
        encoding="utf-8",
    )

    loaded_config = load_streamlit_config(config_path)

    assert loaded_config["simulation"]["population_size"] == 4096
    assert loaded_config["simulation"]["empty_ratio"] == 0.2
    assert loaded_config["simulation"]["property_values_enabled"] is True
    assert loaded_config["teams"]["1"]["population_share"] == 0.8
    assert loaded_config["teams"]["1"]["threshold_mean"] == 0.4


def test_normalize_teams_distribution_returns_unit_sum() -> None:
    """Population shares should be normalized before map generation."""
    distribution = normalize_teams_distribution({1: 20.0, 2: 30.0, 3: 50.0})

    assert distribution == {1: 0.2, 2: 0.3, 3: 0.5}
    assert sum(distribution.values()) == 1.0


def test_schelling_run_moves_agent_only_into_affordable_house() -> None:
    """Unhappy agents should move only to affordable empty houses."""
    schelling = Schelling(
        size=4,
        empty_ratio=0.75,
        n_neighbors=1,
        load_map=False,
        races=2,
        teams_distribution={1: 1.0, 2: 0.0},
        similarity_threshold_distributions={1: (0.2, 0.0), 2: (0.2, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
        pairwise_cultural_multipliers={(1, 2): 1.0},
    )
    schelling.city.set_map(np.array([[1, 0], [0, 0]]))
    schelling.city.instantiate_city(
        MENTAL_VALUES_MAP,
        agent_values_std=0.0,
        similarity_threshold_distributions={1: (0.2, 0.0), 2: (0.2, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
    )
    schelling.city.city[0, 0].property_value = 20.0
    schelling.city.city[0, 1].property_value = 10.0
    schelling.city.city[1, 0].property_value = 13.0
    schelling.city.city[1, 1].property_value = 14.0

    schelling.run()

    assert schelling.city.city[0, 0].agent is None
    assert schelling.city.city[0, 1].agent is not None
    assert schelling.city.city[1, 0].agent is None
    assert schelling.city.city[1, 1].agent is None


def test_schelling_run_keeps_agent_in_place_without_affordable_house() -> None:
    """Unhappy agents should remain in place when no affordable house exists."""
    schelling = Schelling(
        size=4,
        empty_ratio=0.75,
        n_neighbors=1,
        load_map=False,
        races=2,
        teams_distribution={1: 1.0, 2: 0.0},
        similarity_threshold_distributions={1: (0.2, 0.0), 2: (0.2, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
        pairwise_cultural_multipliers={(1, 2): 1.0},
    )
    schelling.city.set_map(np.array([[1, 0], [0, 0]]))
    schelling.city.instantiate_city(
        MENTAL_VALUES_MAP,
        agent_values_std=0.0,
        similarity_threshold_distributions={1: (0.2, 0.0), 2: (0.2, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
    )
    schelling.city.city[0, 0].property_value = 20.0
    schelling.city.city[0, 1].property_value = 13.0
    schelling.city.city[1, 0].property_value = 14.0
    schelling.city.city[1, 1].property_value = 15.0

    schelling.run()

    assert schelling.city.city[0, 0].agent is not None


def test_schelling_run_ignores_house_cost_when_property_values_are_disabled() -> None:
    """Disabling property values should remove economic unhappiness."""
    schelling = Schelling(
        size=4,
        empty_ratio=0.25,
        n_neighbors=1,
        load_map=False,
        races=2,
        teams_distribution={1: 0.5, 2: 0.5},
        similarity_threshold_distributions={1: (0.1, 0.0), 2: (0.1, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
        pairwise_cultural_multipliers={(1, 2): 1.0},
        property_values_enabled=False,
    )
    schelling.city.set_map(np.array([[1, 0], [2, 2]]))
    schelling.city.instantiate_city(
        MENTAL_VALUES_MAP,
        agent_values_std=0.0,
        similarity_threshold_distributions={1: (0.1, 0.0), 2: (0.1, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
    )
    schelling.city.city[0, 0].property_value = 100.0

    schelling.run()

    assert schelling.city.city[0, 0].agent is not None
    assert schelling.city.city[0, 0].agent.team_id == 1


def test_schelling_run_allows_unaffordable_move_when_property_values_are_disabled() -> None:
    """Disabling property values should remove affordability filtering for moves."""
    schelling = Schelling(
        size=4,
        empty_ratio=0.25,
        n_neighbors=1,
        load_map=False,
        races=2,
        teams_distribution={1: 0.5, 2: 0.5},
        similarity_threshold_distributions={1: (0.95, 0.0), 2: (0.1, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
        pairwise_cultural_multipliers={(1, 2): 1.0},
        property_values_enabled=False,
    )
    schelling.city.set_map(np.array([[1, 0], [2, 2]]))
    schelling.city.instantiate_city(
        MENTAL_VALUES_MAP,
        agent_values_std=0.0,
        similarity_threshold_distributions={1: (0.95, 0.0), 2: (0.1, 0.0)},
        income_distributions={1: (1.0, 0.0), 2: (1.0, 0.0)},
    )
    schelling.city.city[0, 1].property_value = 100.0

    schelling.run()

    assert schelling.city.city[0, 1].agent is not None
    assert schelling.city.city[0, 1].agent.team_id == 1
