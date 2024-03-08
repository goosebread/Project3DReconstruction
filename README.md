# 3D Reconstruction Project

## Setup

To install this project and its dependencies, run
```shell
$ pip install -e .
```

## Development setup

(Optional) Install [poetry](https://python-poetry.org/) to manage dependencies:
```shell
$ pipx install poetry  # Install poetry if not already installed.
$ poetry install       # Install this project + development dependencies.
```

(Optional) Install [pre-commit](https://pre-commit.com/) hooks to format and lint
before each commit:
```shell
$ pre-commit install
```

## Tests

Tests can be run using [`pytest`](https://docs.pytest.org):
```shell
$ pytest
```
