"""Command-line interface. Subcommands (diff, label, calibrate) are added as they are built."""

from pathlib import Path
from typing import Annotated

import typer

from runthrough import __version__

app = typer.Typer(
    name="runthrough",
    help="Evaluate deployed chat agents with simulated users.",
    no_args_is_help=True,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"runthrough {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Evaluate deployed chat agents with simulated users."""


@app.command()
def info() -> None:
    """Show the installed versions of runthrough and Inspect AI."""
    from importlib.metadata import version

    typer.echo(f"runthrough {__version__}")
    typer.echo(f"inspect-ai {version('inspect-ai')}")


@app.command()
def validate(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Scenario files or directories to check.", show_default=False),
    ],
) -> None:
    """Check scenario files and list the scenarios they define."""
    from rich.console import Console
    from rich.table import Table

    from runthrough.scenario import ScenarioError, load_scenarios

    console = Console()
    try:
        loaded = load_scenarios(paths)
    except ScenarioError as exc:
        for problem in exc.problems:
            console.print(f"[red]✗[/red] {problem}", highlight=False, soft_wrap=True)
        noun = "problem" if len(exc.problems) == 1 else "problems"
        console.print(f"\n[red]{len(exc.problems)} {noun} found.[/red]")
        raise typer.Exit(code=1) from None

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("id", no_wrap=True)
    table.add_column("turns", justify="right")
    table.add_column("expects", overflow="fold")
    table.add_column("rubrics", justify="right")
    table.add_column("tags")
    for s in loaded:
        expects = [f"+{t}" for t in s.expect.tools_called]
        expects += [f"-{t}" for t in s.expect.tools_not_called]
        if s.expect.tool_order:
            expects.append(" > ".join(s.expect.tool_order))
        if s.expect.language:
            expects.append(f"lang={s.expect.language}")
        table.add_row(
            s.id, str(s.max_turns), ", ".join(expects), str(len(s.rubrics)), ", ".join(s.tags)
        )
    console.print(table)
    noun = "scenario" if len(loaded) == 1 else "scenarios"
    console.print(f"\n[green]✓[/green] {len(loaded)} {noun} valid.")
