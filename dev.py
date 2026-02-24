#!/usr/bin/env -S poetry run python

import glob
import os
import shutil
import click
import sh


def python_source_files():
    include_paths = (
        glob.glob("*.py")
        + glob.glob("lacecore/*.py")
        + glob.glob("lacecore/**/*.py")
        + ["doc/"]
    )
    exclude_paths = []
    return [x for x in include_paths if x not in exclude_paths]


@click.group()
def cli():
    pass


@cli.command()
def install():
    sh.poetry(
        "install",
        "--sync",
        "--extras",
        "doc",
        "--extras",
        "lint",
        "--extras",
        "format",
        "--extras",
        "test",
        "--extras",
        "cli",
        "--extras",
        "obj",
        _fg=True,
    )


@cli.command()
def test():
    sh.pytest(_fg=True)


@cli.command()
def coverage():
    sh.pytest("--cov=lacecore", _fg=True)


@cli.command()
def coverage_report():
    sh.coverage("html", "--fail-under=0", _fg=True)
    sh.open("htmlcov/index.html", _fg=True)


@cli.command()
def lint():
    sh.flake8(*python_source_files(), _fg=True)


@cli.command()
def black():
    sh.black(*python_source_files(), _fg=True)


@cli.command()
def black_check():
    sh.black("--check", *python_source_files(), _fg=True)


@cli.command()
def doc():
    sh.rm("-rf", "build/", "doc/build/", "doc/api/", _fg=True)
    sh.Command("sphinx-build")("-W", "-b", "singlehtml", "doc", "doc/build", _fg=True)


@cli.command()
def doc_open():
    sh.open("doc/build/index.html", _fg=True)


@cli.command()
def clean():
    sh.find(".", "-name", "*.pyc", "-delete", _fg=True)
    sh.find(".", "-name", "__pycache__", "-delete", _fg=True)


@cli.command()
def publish():
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)
    sh.poetry("build", _fg=True)
    sh.twine("upload", *glob.glob("dist/*"), _fg=True)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    cli()
