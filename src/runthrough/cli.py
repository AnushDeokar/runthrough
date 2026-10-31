"""Command-line interface. Subcommands (diff, label, calibrate) are added as they are built."""

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
