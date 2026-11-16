import typer

import echo.cli


def test_cli_app_is_importable():
    assert isinstance(echo.cli.app, typer.Typer)
