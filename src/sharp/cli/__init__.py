"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import equirect_to_cubefaces, predict, render


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(equirect_to_cubefaces.equirect_to_cubefaces_cli, "equirect-to-cubefaces")
