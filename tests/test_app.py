"""Unit tests for application entrypoints."""

from schelling_mod.app import build_parser


def test_build_parser_reads_expected_cli_values() -> None:
    """The CLI parser should expose the documented simulation options."""
    args = build_parser().parse_args(
        [
            "--run_simulation",
            "--population_size",
            "81",
            "--empty_ratio",
            "0.3",
            "--similarity_threshold",
            "0.6",
            "--iterations",
            "4",
        ]
    )

    assert args.run_simulation is True
    assert args.population_size == 81
    assert args.empty_ratio == 0.3
    assert args.similarity_threshold == 0.6
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
