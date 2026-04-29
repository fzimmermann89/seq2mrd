"""Helpers for reading Bruker JCAMP-DX parameter files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_raw_data_path(path: str | Path) -> Path:
    """Resolve a Bruker dataset path to a raw data file."""
    resolved_path = Path(path)
    if not resolved_path.exists() or resolved_path.is_file():
        return resolved_path

    for name in ('fid', 'rawdata.job0'):
        candidate = resolved_path / name
        if candidate.is_file():
            return candidate

    scan_directories = [
        child
        for child in sorted(resolved_path.iterdir())
        if child.is_dir() and child.name.isdigit() and any((child / name).exists() for name in ('fid', 'rawdata.job0'))
    ]
    if len(scan_directories) == 1:
        return resolve_raw_data_path(scan_directories[0])

    raise FileNotFoundError(f'Could not find a Bruker raw data file under {resolved_path}.')


def load_parameter_file(path: Path) -> object | None:
    """Load one Bruker JCAMP-DX parameter file."""
    try:
        from brukerapi.jcampdx import JCAMPDX
    except ImportError as exc:
        raise ImportError('Bruker support requires the brukerapi package to parse JCAMP-DX parameter files.') from exc
    if not path.is_file():
        return None
    return JCAMPDX(path)


def load_parameter_files(raw_path: str | Path) -> dict[str, object | None]:
    """Load the Bruker parameter files associated with a raw data file."""
    resolved_raw_path = resolve_raw_data_path(raw_path)
    dataset_directory = resolved_raw_path.parent
    subject_directory = dataset_directory.parent
    processed_directory = dataset_directory / 'pdata' / '1'

    return {
        'acqp': load_parameter_file(dataset_directory / 'acqp'),
        'method': load_parameter_file(dataset_directory / 'method'),
        'subject': load_parameter_file(subject_directory / 'subject'),
        'visu_pars': load_parameter_file(processed_directory / 'visu_pars'),
        'reco': load_parameter_file(processed_directory / 'reco'),
    }


def first_value(parameter_files: dict[str, object | None], *names: str) -> object | None:
    """Read the first available Bruker parameter value."""
    for name in names:
        for parameter_file in parameter_files.values():
            if parameter_file is None:
                continue
            try:
                value = parameter_file.get_value(name)
            except Exception:  # noqa: BLE001
                value = None
            if value is not None:
                return value
    return None


def first_array(parameter_files: dict[str, object | None], *names: str, dtype: np.dtype | None = None) -> np.ndarray | None:
    """Read the first available Bruker parameter as a NumPy array."""
    value = first_value(parameter_files, *names)
    if value is None:
        return None
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def first_int(parameter_files: dict[str, object | None], *names: str) -> int | None:
    """Read the first available Bruker parameter as an integer."""
    value = first_value(parameter_files, *names)
    if value is None:
        return None
    value_array = np.asarray(value)
    if value_array.size == 0:
        return None
    return int(value_array.reshape(-1)[0])


def first_float(parameter_files: dict[str, object | None], *names: str) -> float | None:
    """Read the first available Bruker parameter as a float."""
    value = first_value(parameter_files, *names)
    if value is None:
        return None
    value_array = np.asarray(value)
    if value_array.size == 0:
        return None
    return float(value_array.reshape(-1)[0])


def first_str(parameter_files: dict[str, object | None], *names: str) -> str | None:
    """Read the first available Bruker parameter as a string."""
    value = first_value(parameter_files, *names)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip('<>')
    value_array = np.asarray(value)
    if value_array.size == 0:
        return None
    return str(value_array.reshape(-1)[0]).strip('<>')
