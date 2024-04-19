#!/usr/bin/env python3
"""
Download datasets.
"""
# References
# ==========
# - https://stackoverflow.com/a/2030027
# - https://stackoverflow.com/q/40544123

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fnmatch
import hashlib
import pathlib
import sys
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Final, NoReturn

import tqdm

if TYPE_CHECKING:
    import http.client
    from collections.abc import Collection, Iterable


@dataclass
class Dataset:
    identifier: str
    """Dataset identifier, used for filtering datasets in the CLI."""

    destination: str
    """Destination path for the dataset, relative to the data directory root."""

    url: str
    """URL to download the dataset from."""

    sha256: bytes | None = None
    """SHA256 checksum of the dataset file."""

    # TODO: add unpacker?


DATASETS: Final = [
    # From https://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html
    Dataset(
        identifier="lund-statlib-images",
        destination="lund/StatueOfLiberty/StatLib.zip",
        url="http://vision.maths.lth.se/calledataset/StatLib/StatLib.zip",
        sha256=bytes.fromhex(
            "04caaf7a1b9d972ebee0ae05f0e5bfc4ab60a02c0463ba9d73e6e7e679710793"
        ),
    ),
    Dataset(
        identifier="lund-statlib-points",
        destination="lund/StatueOfLiberty/data.mat",
        url="http://vision.maths.lth.se/calledataset/StatLib/data.mat",
        sha256=bytes.fromhex(
            "4c3e196e389f097fc34b7f412eccd8e880c370c5527bad171f885f902cb2dca3"
        ),
    ),
    # From https://gruvi-3dv.cs.sfu.ca/ocrtoc-3d-reconstruction/ (https://arxiv.org/abs/2203.11397)
    # Dataset(
    #     identifier="shrestha-scene-bottle",
    #     destination="Datasets/shrestha/bottle.zip",
    #     url="https://gruvi-3dv.cs.sfu.ca/ocrtoc-3d-reconstruction/bottle.zip",
    # ),
    # Dataset(
    #     identifier="shrestha-scene-bowl",
    #     destination="Datasets/shrestha/bowl.zip",
    #     url="https://gruvi-3dv.cs.sfu.ca/ocrtoc-3d-reconstruction/bowl.zip",
    # ),
]


def _make_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        help="Directory to store datasets to. Defaults to the 'Data' directory"
        " in the project root.",
    )
    parser.add_argument(
        "--max-jobs", type=int, help="Maximum number of concurrent download jobs."
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--list", action="store_true", help="List available datasets."
    )
    action_group.add_argument(
        "--download",
        metavar="FILTER",
        help="Download datasets. Accepts a glob pattern to filter which datasets to"
        " download. Pass '*' to download all datasets",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts.")

    return parser


def _find_project_root(search_base: pathlib.Path) -> pathlib.Path:
    """Locate project root directory for the given path."""
    curr_dir = search_base
    while True:
        if curr_dir.joinpath("pyproject.toml").is_file():
            return curr_dir
        next_dir = curr_dir.parent
        if next_dir == curr_dir:
            raise RuntimeError(f"unable to find project root from '{search_base}'")
        curr_dir = next_dir


def _download_url(
    url: str,
    file: BinaryIO,
    bar: tqdm.tqdm[NoReturn],
    chunk_size: int = 16 * 1024,
) -> bytes:
    """Download URL resource to the given file object with progress bar."""
    digest = hashlib.sha256(usedforsecurity=False)

    response: http.client.HTTPResponse
    with urllib.request.urlopen(url) as response:
        total_size = int(response.info()["Content-Length"].strip())
        bar.set_description(bar.desc.split(" ")[0])
        bar.reset(total_size)

        while chunk := response.read(chunk_size):
            file.write(chunk)
            # GIL released when chunk size is larger than 2047 bytes.
            digest.update(chunk)
            bar.update(len(chunk))

    return digest.digest()


def _run_download_datasets(
    datasets: list[Dataset], data_dir: pathlib.Path, num_workers: int | None
) -> list[str]:
    """Download the given datasets in parallel with stdout progress bars."""
    failures: list[str] = []
    with (
        contextlib.ExitStack() as bar_stack,
        concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor,
    ):
        future_to_dataset: dict[
            concurrent.futures.Future[bytes], tuple[Dataset, tqdm.tqdm[NoReturn]]
        ] = {}
        for index, ds in enumerate(datasets):
            # Setup progress bar.
            bar = tqdm.tqdm(
                position=index,
                desc=f"{ds.identifier} (waiting)",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=True,
            )
            # TODO: maybe fix bar order reversing on close.
            bar_stack.enter_context(contextlib.closing(bar))

            # Ensure destination directory exists.
            destination = data_dir.joinpath(ds.destination)
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Submit future.
            future = executor.submit(
                _download_url,
                url=ds.url,
                file=destination.open("wb"),
                bar=bar,
            )
            future_to_dataset[future] = ds, bar

        # Collect results.
        for future in concurrent.futures.as_completed(future_to_dataset):
            finished_ds, finished_bar = future_to_dataset[future]
            try:
                digest = future.result()
            except Exception as ex:  # noqa: BLE001
                finished_bar.set_description(f"{finished_ds.identifier}: ERROR! {ex}")
                finished_bar.refresh()
                failures.append(finished_ds.identifier)
                continue

            # Verify checksum if available.
            if finished_ds.sha256 and digest != finished_ds.sha256:
                finished_bar.set_description(
                    f"{finished_ds.identifier}: WARNING! failed checksum -"
                    f" expected {finished_ds.sha256.hex()[:8]},"
                    f" got {digest.hex()[:8]}"
                )
                finished_bar.refresh()
                failures.append(finished_ds.identifier)
                continue

            finished_bar.set_description(f"{finished_ds.identifier} (done)")
            finished_bar.refresh()

    return failures


def _display_datasets(datasets: Iterable[Dataset], data_dir: pathlib.Path) -> None:
    """Display the given datasets to stdout."""
    for ds in datasets:
        exists = data_dir.joinpath(ds.destination).exists()
        print(f" - {ds.identifier} ({'*' if exists else ' '}): {ds.url}")


def _prompt_confirmation(prompt: str, ok_set: Collection[str]) -> bool:
    """Prompt the user and return True if the response is found in the ok_set."""
    response = input(prompt)
    return response.strip() in ok_set


def main(cli_args: list[str]) -> None:
    """CLI entrypoint."""
    parser = _make_parser()
    args = parser.parse_args(cli_args)

    # Resolve data directory.
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = _find_project_root(pathlib.Path.cwd()).joinpath("data")
    data_dir.mkdir(parents=False, exist_ok=True)
    print(f"Data directory: {data_dir}")

    # Resolve number of workers.
    num_workers = args.max_jobs
    if num_workers is not None and num_workers < -1:
        num_workers = None  # use ThreadPoolExecutor default

    # Handle --list
    if args.list:
        print("Available datasets (* if already downloaded):")
        _display_datasets(DATASETS, data_dir)
        return

    # Handle --download
    if args.download:
        print(f"Filter: {args.download}")

        requested_datesets = [
            ds for ds in DATASETS if fnmatch.fnmatch(ds.identifier, args.download)
        ]

        print(
            f"Selected {len(requested_datesets)}/{len(DATASETS)}"
            f" datasets (* if already downloaded):"
        )
        _display_datasets(requested_datesets, data_dir)

        if not (
            args.yes
            or _prompt_confirmation(
                "Proceed with download (y/n)? ", ["y", "Y", "yes", "YES"]
            )
        ):
            print("Cancelled")
            return

        print("Downloading...")
        failures = _run_download_datasets(requested_datesets, data_dir, num_workers)
        print("Finished downloads")
        if failures:
            print(f"ERROR! {len(failures)} download(s) were unsuccessful: {failures}")
            sys.exit(1)
        print(f"All {len(requested_datesets)} downloads were successful.")
        return

    # No action was specified.

    parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
