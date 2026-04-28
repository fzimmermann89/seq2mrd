"""Command line interface for seq2mrd."""

from __future__ import annotations

import argparse
from pathlib import Path

from seq2mrd.convenience import convert_bruker, convert_ge, convert_numpy, convert_siemens
from seq2mrd.skeleton import MRDSkeleton


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(prog='seq2mrd')
    subparsers = parser.add_subparsers(dest='vendor', required=True)

    for vendor_name in ('siemens', 'bruker', 'ge', 'numpy'):
        vendor_parser = subparsers.add_parser(vendor_name)
        vendor_parser.add_argument('--raw', required=True)
        vendor_parser.add_argument('--seq', required=True)
        vendor_parser.add_argument('--output', required=True)
        vendor_parser.add_argument('--dataset-name', default='dataset')

    return parser


def main() -> None:
    """Run the command line interface."""
    arguments = _build_argument_parser().parse_args()

    conversion_function_by_vendor = {
        'siemens': convert_siemens,
        'bruker': convert_bruker,
        'ge': convert_ge,
        'numpy': convert_numpy,
    }
    convert_vendor = conversion_function_by_vendor[arguments.vendor]
    header, acquisitions = convert_vendor(arguments.seq, arguments.raw)
    MRDSkeleton(header=header, acquisitions=list(acquisitions)).write(
        Path(arguments.output),
        dataset_name=arguments.dataset_name,
    )
