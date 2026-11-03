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


LogsArg = Annotated[
    list[Path], typer.Argument(help="Inspect .eval logs or folders of logs.", show_default=False)
]
LabelsOpt = Annotated[
    Path, typer.Option("--labels", "-l", help="Where your labels are stored (JSON Lines).")
]
ScorerOpt = Annotated[
    str | None, typer.Option("--scorer", "-s", help="Only this rubric scorer, e.g. tone.")
]


@app.command()
def label(
    logs: LogsArg,
    labels: LabelsOpt = Path("labels.jsonl"),
    scorer: ScorerOpt = None,
    port: Annotated[int, typer.Option(help="Port for the local labelling page.")] = 7700,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the browser.")
    ] = True,
) -> None:
    """Label judged conversations yourself, in a local web page."""
    import webbrowser

    from runthrough.calibrate import LabelStore, collect_items
    from runthrough.calibrate.server import make_server

    items = collect_items(logs, scorer=scorer)
    if not items:
        typer.echo("No rubric-judge scores found in these logs.", err=True)
        raise typer.Exit(code=1)
    store = LabelStore(labels)
    server = make_server(items, store, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    done = sum(item.key in store.load() for item in items)
    typer.echo(f"{len(items)} judged conversations, {done} already labelled.")
    typer.echo(f"Labelling page: {url}  (labels saved to {labels}; Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    finally:
        server.server_close()


@app.command()
def calibrate(
    logs: LogsArg,
    labels: LabelsOpt = Path("labels.jsonl"),
    scorer: ScorerOpt = None,
    show: Annotated[int, typer.Option(help="Disagreements to list per rubric.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Report how often each rubric judge agrees with your labels."""
    import json

    from rich.console import Console
    from rich.table import Table

    from runthrough.calibrate import LabelStore, agreement_by_scorer, collect_items

    results = agreement_by_scorer(collect_items(logs, scorer=scorer), LabelStore(labels).load())
    if as_json:
        payload = [
            {**r.model_dump(), "agreement": r.agreement, "strength": r.strength} for r in results
        ]
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    console = Console()
    if not results:
        console.print("No rubric-judge scores found in these logs.")
        raise typer.Exit(code=1)

    table = Table(show_edge=False, pad_edge=False)
    for column, justify in [
        ("rubric", "left"),
        ("labelled", "right"),
        ("agree", "right"),
        ("kappa", "right"),
        ("strength", "left"),
        ("judge ✓ you ✗", "right"),
        ("judge ✗ you ✓", "right"),
    ]:
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    for r in results:
        table.add_row(
            r.scorer,
            f"{r.labeled}/{r.judged}",
            "-" if r.agreement is None else f"{r.agreement:.0%}",
            "-" if r.kappa is None else f"{r.kappa:.2f}",
            r.strength,
            str(r.judge_pass_you_fail),
            str(r.judge_fail_you_pass),
        )
    console.print(table)

    for r in results:
        if r.unreadable:
            console.print(
                f"\n[yellow]{r.scorer}[/yellow]: {r.unreadable} unreadable judge replies."
            )
        if r.disagreements and show > 0:
            console.print(f"\n[bold]{r.scorer}[/bold]: disagreements")
            for d in r.disagreements[:show]:
                console.print(
                    f"  {d.sample_id} (epoch {d.epoch}): judge {d.judge}, you {d.you}",
                    highlight=False,
                )
                if d.judge_reasoning:
                    console.print(f"    judge: {d.judge_reasoning}", highlight=False, style="dim")
                if d.note:
                    console.print(f"    you: {d.note}", highlight=False, style="dim")
    if any(r.labeled < 30 for r in results):
        console.print(
            "\n[dim]Fewer than 30 labels for a rubric gives a rough estimate; label more before "
            "relying on it.[/dim]"
        )
