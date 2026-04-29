"""Generic Bruker fid decoding."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from seq2mrd.raw.bruker.parameters import first_array, first_int, first_value, resolve_raw_data_path


def decode_fid(raw_path: str | Path, parameter_files: dict[str, object]) -> np.ndarray:
    """Decode Bruker raw data into `(n_adcs, n_coils, n_readout)` order."""
    resolved_raw_path = resolve_raw_data_path(raw_path)
    acq_size = first_array(parameter_files, 'ACQ_size', dtype=np.int64)
    if acq_size is None:
        raise ValueError('Missing Bruker parameter ACQ_size.')
    acq_size = acq_size.reshape(-1)

    channels = first_int(parameter_files, 'PVM_EncNReceivers') or 1
    n_objects = first_int(parameter_files, 'NI') or 1
    n_repetitions = first_int(parameter_files, 'NR') or 1
    n_segments = max(first_int(parameter_files, 'PVM_EpiNShots', 'NSegments') or 1, 1)
    n_interleaves = first_int(parameter_files, 'PVM_SpiralNbOfInterleaves')
    n_projections = first_int(parameter_files, 'NPro')
    phase_factor = max(first_int(parameter_files, 'ACQ_phase_factor') or 1, 1)
    encoded_matrix = first_array(parameter_files, 'PVM_EncMatrix', 'PVM_Matrix', dtype=np.int64)
    encoded_matrix = encoded_matrix.reshape(-1) if encoded_matrix is not None else np.empty(0, dtype=np.int64)

    raw_format = str(first_value(parameter_files, 'GO_raw_data_format') or '')
    byte_order = str(first_value(parameter_files, 'BYTORDA') or '')
    dtype_map = {
        ('GO_32BIT_SGN_INT', 'little'): np.dtype('int32').newbyteorder('<'),
        ('GO_16BIT_SGN_INT', 'little'): np.dtype('int16').newbyteorder('<'),
        ('GO_32BIT_FLOAT', 'little'): np.dtype('float32').newbyteorder('<'),
        ('GO_32BIT_SGN_INT', 'big'): np.dtype('int32').newbyteorder('>'),
        ('GO_16BIT_SGN_INT', 'big'): np.dtype('int16').newbyteorder('>'),
        ('GO_32BIT_FLOAT', 'big'): np.dtype('float32').newbyteorder('>'),
    }
    dtype = dtype_map.get((raw_format, byte_order))
    if dtype is None:
        word_size = str(first_value(parameter_files, 'ACQ_word_size') or '')
        if word_size == '_32_BIT' and byte_order == 'little':
            dtype = np.dtype('int32').newbyteorder('<')
        elif word_size == '_32_BIT' and byte_order == 'big':
            dtype = np.dtype('int32').newbyteorder('>')
        elif word_size == '_16_BIT' and byte_order == 'little':
            dtype = np.dtype('int16').newbyteorder('<')
        elif word_size == '_16_BIT' and byte_order == 'big':
            dtype = np.dtype('int16').newbyteorder('>')
        else:
            raise ValueError('Unsupported Bruker raw data format.')

    if str(first_value(parameter_files, 'GO_block_size') or '') == 'Standard_KBlock_Format':
        block_size = int(math.ceil(int(acq_size[0]) * channels * dtype.itemsize / 1024) * 1024 / dtype.itemsize)
    else:
        block_size = int(acq_size[0]) * channels
    block_count = resolved_raw_path.stat().st_size // (block_size * dtype.itemsize)

    is_segmented_readout = encoded_matrix.size >= 2 and n_segments > 1 and int(encoded_matrix[0]) != int(acq_size[0] // 2)
    has_reversed_odd_lines = is_segmented_readout and first_int(parameter_files, 'PVM_EpiNShots', 'NSegments') is not None

    if is_segmented_readout:
        n_k0 = int(encoded_matrix[0])
        n_k1 = int(encoded_matrix[1])
        n_k2 = 1
        explicit_extra_sizes = tuple(int(size) for size in acq_size[2:] if int(size) > 1)
    else:
        n_k0 = int(acq_size[0] // 2)
        n_k1 = int(n_interleaves or n_projections or (acq_size[1] if acq_size.size >= 2 else 1))
        n_k2 = int(acq_size[2]) if acq_size.size >= 3 and not (n_interleaves or n_projections) else 1
        explicit_extra_sizes = tuple(int(size) for size in acq_size[3:] if int(size) > 1)

    if is_segmented_readout:
        base_count = n_segments * n_objects * n_repetitions * int(np.prod(explicit_extra_sizes or (1,)))
    else:
        base_count = n_k1 * n_k2 * n_objects * n_repetitions * int(np.prod(explicit_extra_sizes or (1,)))
    if base_count <= 0 or block_count <= 0 or block_count % base_count != 0:
        raise ValueError('Could not infer a generic Bruker raw data layout.')

    extra_sizes = explicit_extra_sizes
    remaining_count = block_count // base_count
    if remaining_count > 1:
        extra_sizes = (*extra_sizes, int(remaining_count))

    raw = np.array(np.memmap(resolved_raw_path, mode='r', dtype=dtype, shape=(block_size, block_count), order='F'))
    dig_np = first_int(parameter_files, 'PVM_DigNp')
    if is_segmented_readout and dig_np is not None:
        acquisition_length = int(2 * dig_np * channels)
    else:
        acquisition_length = int(acq_size[0]) * channels
    if block_size > acquisition_length:
        offset = block_size - acquisition_length if is_segmented_readout else 0
        raw = raw[offset : offset + acquisition_length, :]
    else:
        raw = raw[:acquisition_length, :]

    complex_data = raw[0::2, ...] + 1j * raw[1::2, ...]

    if is_segmented_readout:
        encoding_space = (
            n_k0 * n_k1 // n_segments,
            channels,
            n_segments,
            n_objects,
            n_repetitions,
            *extra_sizes,
        )
        permute = (0, 2, 3, 4, *range(5, 5 + len(extra_sizes)), 1)
        k_space = (n_k0, n_k1, n_objects, n_repetitions, *extra_sizes, channels)
    else:
        encode_inner = phase_factor if n_k1 % phase_factor == 0 else 1
        encode_outer = n_k1 // encode_inner
        if n_k2 > 1:
            encoding_space = (
                n_k0,
                channels,
                encode_inner,
                n_objects,
                encode_outer,
                n_k2,
                n_repetitions,
                *extra_sizes,
            )
            permute = (0, 2, 4, 5, 3, 6, *range(7, 7 + len(extra_sizes)), 1)
            k_space = (n_k0, n_k1, n_k2, n_objects, n_repetitions, *extra_sizes, channels)
        else:
            encoding_space = (
                n_k0,
                channels,
                encode_inner,
                n_objects,
                encode_outer,
                n_repetitions,
                *extra_sizes,
            )
            permute = (0, 2, 4, 3, 5, *range(6, 6 + len(extra_sizes)), 1)
            k_space = (n_k0, n_k1, n_objects, n_repetitions, *extra_sizes, channels)

    data = np.reshape(complex_data, encoding_space, order='F')
    data = np.transpose(data, permute)
    data = np.reshape(data, k_space, order='F')

    encoding_steps = first_array(parameter_files, 'PVM_EncSteps1', dtype=np.int64)
    if encoding_steps is not None:
        encoding_steps = encoding_steps.reshape(-1)
        if encoding_steps.size == data.shape[1]:
            order = np.argsort(encoding_steps)
            if not np.array_equal(order, encoding_steps):
                data = data.copy()
                data[:, :, ...] = data[:, order, ...]

    if has_reversed_odd_lines:
        data = data.copy()
        data[:, 1::2, ...] = data[::-1, 1::2, ...]

    pre_size = max(first_int(parameter_files, 'PVM_TrajPreSize') or 0, 0)
    post_size = max(first_int(parameter_files, 'PVM_TrajPostSize') or 0, 0)
    target_samples = first_int(parameter_files, 'PVM_TrajResultSize')
    if target_samples is not None and data.shape[0] != target_samples:
        if pre_size or post_size:
            if data.shape[0] - pre_size - post_size == target_samples:
                data = data[pre_size : data.shape[0] - post_size if post_size else data.shape[0], ...]
            else:
                raise ValueError(f'Bruker readout has {data.shape[0]} samples, but expected {target_samples}.')
        else:
            raise ValueError(f'Bruker readout has {data.shape[0]} samples, but expected {target_samples}.')

    data = np.moveaxis(data, -1, -2)
    return np.reshape(data, (-1, data.shape[-2], data.shape[-1]), order='C')
