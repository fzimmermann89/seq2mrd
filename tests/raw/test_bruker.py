"""Tests for the Bruker raw reader."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from seq2mrd import BrukerRaw, MRDSkeleton


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


def test_brukerraw_fills_skeleton(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bruker raw data should fill a pulseq-derived skeleton."""
    sequence_path = build_test_sequence_file(tmp_path / 'sequence.seq')

    fake_parameter_files = {'subject': object(), 'visu_pars': object(), 'acqp': object(), 'method': object(), 'reco': object()}
    fake_data = np.array(
        [
            [[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j]],
            [[9 + 9j, 10 + 10j, 11 + 11j, 12 + 12j, 13 + 13j, 14 + 14j, 15 + 15j, 16 + 16j]],
        ],
        dtype=np.complex64,
    )

    def fake_load_parameter_files(_raw_path: object) -> dict[str, object]:
        return fake_parameter_files

    def fake_decode_fid(_raw_path: object, _parameter_files: object) -> np.ndarray:
        return fake_data

    def fake_first_str(_parameter_files: object, *names: str) -> str | None:
        if names == ('SUBJECT_name_string', 'VisuSubjectName', 'SUBJECT_id'):
            return 'TestSubject'
        if names == ('ACQ_time',):
            return '<2026-04-29T12:34:56,000000+0000>'
        if names == ('SUBJECT_position',):
            return 'Head_Prone'
        return None

    def fake_first_float(_parameter_files: object, *names: str) -> float | None:
        if names == ('PVM_FrqWork', 'PVM_FrqRef', 'VisuAcqImagingFrequency'):
            return 127.7
        return None

    def fake_first_array(
        _parameter_files: object,
        *names: str,
        dtype: np.dtype | None = None,
    ) -> np.ndarray | None:
        if names == ('VisuCoreOrientation',):
            array = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        elif names == ('VisuCorePosition',):
            array = np.array(
                [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                ],
                dtype=np.float32,
            )
        elif names == ('VisuCoreExtent',):
            array = np.array([8.0, 8.0, 0.0], dtype=np.float32)
        else:
            return None
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def fake_first_int(_parameter_files: object, *names: str) -> int | None:
        if names == ('VisuCoreDim',):
            return 2
        return None

    monkeypatch.setattr('seq2mrd.raw.bruker.source.load_parameter_files', fake_load_parameter_files)
    monkeypatch.setattr('seq2mrd.raw.bruker.source.decode_fid', fake_decode_fid)
    monkeypatch.setattr('seq2mrd.raw.bruker.source.first_str', fake_first_str)
    monkeypatch.setattr('seq2mrd.raw.bruker.source.first_float', fake_first_float)
    monkeypatch.setattr('seq2mrd.raw.bruker.source.first_array', fake_first_array)
    monkeypatch.setattr('seq2mrd.raw.bruker.source.first_int', fake_first_int)

    skeleton = BrukerRaw('dummy-fid')(MRDSkeleton.from_seq(sequence_path))

    assert skeleton.header.subjectInformation.patientName == 'TestSubject'
    assert skeleton.header.acquisitionSystemInformation.systemVendor == 'Bruker'
    assert skeleton.header.experimentalConditions.H1resonanceFrequency_Hz == 127700000
    assert skeleton.header.measurementInformation.patientPosition == 'Head_Prone'
    np.testing.assert_array_equal(skeleton.acquisitions[0].data, fake_data[0])
    np.testing.assert_array_equal(skeleton.acquisitions[1].data, fake_data[1])
    np.testing.assert_allclose(skeleton.acquisitions[0].position, (14.0, 16.0, 30.0))
    np.testing.assert_allclose(skeleton.acquisitions[1].position, (44.0, 46.0, 60.0))
    np.testing.assert_allclose(skeleton.acquisitions[0].read_dir, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(skeleton.acquisitions[0].phase_dir, (0.0, -1.0, 0.0))
    np.testing.assert_allclose(skeleton.acquisitions[0].slice_dir, (0.0, 0.0, -1.0))
