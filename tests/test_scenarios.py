from pathlib import Path

import pytest
from typer.testing import CliRunner

from runthrough.cli import app
from runthrough.scenario import (
    SCENARIO_KEY,
    ScenarioError,
    load_scenarios,
    scenario_from_metadata,
    scenarios,
)

EXAMPLES = Path(__file__).parents[1] / "examples" / "suites" / "support" / "scenarios"

MINIMAL = """
id: greet
persona: A parent
goal: Ask about opening hours
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def problems(paths) -> list[str]:
    with pytest.raises(ScenarioError) as info:
        load_scenarios(paths)
    return info.value.problems


class TestLoading:
    def test_single_scenario_with_defaults(self, tmp_path: Path) -> None:
        [s] = load_scenarios(write(tmp_path, "greet.yaml", MINIMAL))
        assert (s.id, s.max_turns, s.opening, s.rubrics) == ("greet", 8, [], [])
        assert s.source_file == str(tmp_path / "greet.yaml")

    def test_list_file(self, tmp_path: Path) -> None:
        text = "- {id: a, persona: p, goal: g}\n- {id: b, persona: p, goal: g}\n"
        assert [s.id for s in load_scenarios(write(tmp_path, "two.yml", text))] == ["a", "b"]

    def test_directory_is_recursive_and_sorted(self, tmp_path: Path) -> None:
        write(tmp_path, "b/second.yaml", MINIMAL.replace("greet", "second"))
        write(tmp_path, "a/first.yaml", MINIMAL.replace("greet", "first"))
        write(tmp_path, "notes.txt", "ignored")
        assert [s.id for s in load_scenarios(tmp_path)] == ["first", "second"]

    def test_opening_string_becomes_list(self, tmp_path: Path) -> None:
        [s] = load_scenarios(write(tmp_path, "s.yaml", MINIMAL + "opening: Hello\n"))
        assert s.opening == ["Hello"]

    def test_rubrics_resolve_relative_to_file(self, tmp_path: Path) -> None:
        write(tmp_path, "rubrics/tone.md", "# Tone")
        path = write(tmp_path, "scenarios/s.yaml", MINIMAL + "rubrics: [../rubrics/tone.md]\n")
        [s] = load_scenarios(path)
        assert s.rubrics == [str((tmp_path / "rubrics" / "tone.md").resolve())]

    def test_examples_are_valid(self) -> None:
        ids = [s.id for s in load_scenarios(EXAMPLES)]
        assert ids == ["refund-damaged-item", "refund-asks-for-human"]


class TestErrors:
    def test_unknown_field_names_file_and_field(self, tmp_path: Path) -> None:
        path = write(tmp_path, "s.yaml", MINIMAL + "max_turn: 3\n")
        assert problems(path) == [f"{path}: max_turn: Extra inputs are not permitted"]

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        path = write(tmp_path, "s.yaml", "id: x\n")
        assert {p.split(": ")[1] for p in problems(path)} == {"persona", "goal"}

    def test_list_entries_are_indexed(self, tmp_path: Path) -> None:
        path = write(tmp_path, "s.yaml", "- {id: a, persona: p, goal: g}\n- {id: b, persona: p}\n")
        assert problems(path) == [f"{path}[1]: goal: Field required"]

    def test_duplicate_ids_across_files(self, tmp_path: Path) -> None:
        write(tmp_path, "a.yaml", MINIMAL)
        write(tmp_path, "b.yaml", MINIMAL)
        [problem] = problems(tmp_path)
        assert "duplicate id 'greet'" in problem and "a.yaml" in problem

    def test_contradictory_expectations(self, tmp_path: Path) -> None:
        text = MINIMAL + "expect: {tools_called: [book], tools_not_called: [book]}\n"
        [problem] = problems(write(tmp_path, "s.yaml", text))
        assert problem.endswith("expect: book is in both tools_called and tools_not_called")

    def test_opening_longer_than_max_turns(self, tmp_path: Path) -> None:
        text = MINIMAL + "max_turns: 1\nopening: [Hi, Hello]\n"
        [problem] = problems(write(tmp_path, "s.yaml", text))
        assert "opening has 2 messages but max_turns is 1" in problem

    def test_missing_rubric(self, tmp_path: Path) -> None:
        path = write(tmp_path, "s.yaml", MINIMAL + "rubrics: [nope.md]\n")
        assert problems(path) == [f"{path}: rubric file not found: nope.md"]

    def test_bad_id_and_bad_yaml_reported_together(self, tmp_path: Path) -> None:
        write(tmp_path, "a.yaml", MINIMAL.replace("id: greet", "id: has space"))
        write(tmp_path, "b.yaml", "id: [unclosed\n")
        found = problems(tmp_path)
        assert len(found) == 2
        assert "id: String should match pattern" in found[0]
        assert "invalid YAML" in found[1]

    def test_source_file_cannot_be_set(self, tmp_path: Path) -> None:
        [problem] = problems(write(tmp_path, "s.yaml", MINIMAL + "source_file: x\n"))
        assert "source_file is set by the loader" in problem

    def test_empty_file_and_missing_path(self, tmp_path: Path) -> None:
        empty = write(tmp_path, "empty.yaml", "")
        found = problems([empty, tmp_path / "missing"])
        assert found == [
            f"{tmp_path / 'missing'}: file or directory not found",
            f"{empty}: file is empty",
        ]


class TestDataset:
    def test_samples_carry_the_scenario(self) -> None:
        dataset = scenarios(EXAMPLES)
        sample = dataset[0]
        assert sample.id == "refund-damaged-item"
        assert sample.metadata is not None
        restored = scenario_from_metadata(sample.metadata)
        assert restored.expect.tool_order == ["lookup_order", "issue_refund"]
        assert "Goal: Get a refund" in str(sample.input)

    def test_tag_filter(self) -> None:
        assert [s.id for s in scenarios(EXAMPLES, tags=["handoff"])] == ["refund-asks-for-human"]

    def test_missing_scenario_metadata(self) -> None:
        with pytest.raises(ValueError, match="no runthrough scenario"):
            scenario_from_metadata({"other": 1})

    def test_metadata_key_is_stable(self) -> None:
        assert SCENARIO_KEY == "runthrough_scenario"


class TestValidateCommand:
    def test_valid_suite(self) -> None:
        result = CliRunner().invoke(app, ["validate", str(EXAMPLES)])
        assert result.exit_code == 0, result.output
        assert "2 scenarios valid" in result.output

    def test_invalid_suite(self, tmp_path: Path) -> None:
        write(tmp_path, "s.yaml", "id: x\n")
        result = CliRunner().invoke(app, ["validate", str(tmp_path)])
        assert result.exit_code == 1
        assert "2 problems found" in result.output
