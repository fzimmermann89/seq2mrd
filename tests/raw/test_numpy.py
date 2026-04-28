"""Tests for the NumPy raw reader."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from seq2mrd import MRDSkeleton, NumpyRaw


def build_test_sequence_file(path: Path) -> Path:
    """Build and write a small self-contained pulseq test sequence."""
    sequence = pp.Sequence(system=pp.Opts(B0=3))
    readout_gradient = pp.make_trapezoid('x', area=8, duration=1e-3)
    phase_encode_zero = pp.make_trapezoid('y', area=0, duration=1e-3)
    phase_encode_one = pp.make_trapezoid('y', area=1, duration=1e-3)
    rf = pp.make_block_pulse(math.pi / 8, duration=1e-3)

    sequence.add_block(rf)
    sequence.add_block(phase_encode_zero)
    sequence.add_block(
        readout_gradient,
        pp.make_adc(num_samples=8, duration=8e-4, delay=1e-4),
        pp.make_label(label='LIN', type='SET', value=0),
    )
    sequence.add_block(phase_encode_one)
    sequence.add_block(
        readout_gradient,
        pp.make_adc(num_samples=8, duration=8e-4, delay=1e-4),
        pp.make_label(label='LIN', type='SET', value=1),
    )
    sequence.set_definition('FOV', [0.2, 0.2, 0.01])
    sequence.write(str(path))
    return path


def test_numpyraw_fills_skeleton(tmp_path: Path) -> None:
    """NumPy raw data should fill a pulseq-derived skeleton."""
    sequence_path = build_test_sequence_file(tmp_path / 'sequence.seq')
    raw_path = tmp_path / 'raw.npy'
    raw_data = np.array(
        [
            [
                np.arange(8, dtype=np.float32) + 1j * np.arange(8, dtype=np.float32),
                np.arange(8, 16, dtype=np.float32) + 1j * np.arange(8, 16, dtype=np.float32),
            ],
            [
                np.arange(16, 24, dtype=np.float32) + 1j * np.arange(16, 24, dtype=np.float32),
                np.arange(24, 32, dtype=np.float32) + 1j * np.arange(24, 32, dtype=np.float32),
            ],
        ],
        dtype=np.complex64,
    )
    np.save(raw_path, raw_data)

    skeleton = NumpyRaw(raw_path)(MRDSkeleton.from_seq(sequence_path))

    assert skeleton.acquisitions[0].active_channels == 2
    np.testing.assert_array_equal(skeleton.acquisitions[0].data, raw_data[0])
    np.testing.assert_array_equal(skeleton.acquisitions[1].data, raw_data[1])


def test_numpyraw_raises_for_mismatched_readout_count(tmp_path: Path) -> None:
    """NumPy raw data should match the number of pulseq ADC events."""
    sequence_path = build_test_sequence_file(tmp_path / 'sequence.seq')
    raw_path = tmp_path / 'raw.npy'
    np.save(raw_path, np.zeros((1, 2, 8), dtype=np.complex64))

    with pytest.raises(ValueError, match='different number of ADC readouts'):
        _ = NumpyRaw(raw_path)(MRDSkeleton.from_seq(sequence_path))
