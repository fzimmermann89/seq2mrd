"""Tests for pulseq-derived MRD skeleton creation."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pypulseq as pp

from seq2mrd.skeleton import MRDSkeleton


def build_test_sequence_file(
    path: Path,
    *,
    include_labels: bool = True,
    frequency_offsets: tuple[float, ...] = (0.0, 0.0),
    readout_oversampling_factor: float | None = None,
    recon_fov: tuple[float, float, float] | None = None,
    encoding_matrix: tuple[int, int, int] | None = None,
    recon_matrix: tuple[int, int, int] | None = None,
    trajectory_type: str | None = None,
) -> Path:
    """Build and write a small self-contained pulseq test sequence."""
    sequence = pp.Sequence(system=pp.Opts(B0=3))
    readout_gradient = pp.make_trapezoid('x', area=8, duration=1e-3)
    phase_encode_zero = pp.make_trapezoid('y', area=0, duration=1e-3)
    phase_encode_one = pp.make_trapezoid('y', area=1, duration=1e-3)
    rf = pp.make_block_pulse(math.pi / 8, duration=1e-3)

    sequence.add_block(rf)
    sequence.add_block(phase_encode_zero)
    first_adc = pp.make_adc(num_samples=8, duration=8e-4, delay=1e-4, freq_offset=frequency_offsets[0])
    if include_labels:
        sequence.add_block(
            readout_gradient,
            first_adc,
            pp.make_label(label='LIN', type='SET', value=0),
            pp.make_label(label='REP', type='SET', value=0),
        )
    else:
        sequence.add_block(readout_gradient, first_adc)

    sequence.add_block(phase_encode_one)
    second_adc = pp.make_adc(num_samples=8, duration=8e-4, delay=1e-4, freq_offset=frequency_offsets[1])
    if include_labels:
        sequence.add_block(
            readout_gradient,
            second_adc,
            pp.make_label(label='LIN', type='SET', value=1),
            pp.make_label(label='REP', type='SET', value=0),
        )
    else:
        sequence.add_block(readout_gradient, second_adc)

    sequence.set_definition('FOV', [0.2, 0.2, 0.01])
    sequence.set_definition('TR', 0.01)
    sequence.set_definition('TE', 0.002)
    sequence.set_definition('TI', 0.1)

    if recon_fov is not None:
        sequence.set_definition('ReconFOV', list(recon_fov))
    if encoding_matrix is not None:
        sequence.set_definition('EncodingMatrix', list(encoding_matrix))
    if recon_matrix is not None:
        sequence.set_definition('ReconMatrix', list(recon_matrix))
    if trajectory_type is not None:
        sequence.set_definition('TrajectoryType', trajectory_type)
    if readout_oversampling_factor is not None:
        sequence.set_definition('ReadoutOversamplingFactor', readout_oversampling_factor)

    sequence.write(str(path))
    return path


def test_mrdskeleton_from_seq(tmp_path: Path) -> None:
    """from_seq should create a usable MRD skeleton from a labelled pulseq file."""
    sequence_path = build_test_sequence_file(tmp_path / 'sequence.seq', readout_oversampling_factor=2.0)

    skeleton = MRDSkeleton.from_seq(sequence_path)

    assert len(skeleton.acquisitions) == 2
    assert skeleton.acquisitions[0].number_of_samples == 8
    assert skeleton.acquisitions[0].trajectory_dimensions == 2
    assert skeleton.acquisitions[0].sample_time_us == 100.0
    assert skeleton.acquisitions[0].idx.kspace_encode_step_1 == 0
    assert skeleton.acquisitions[1].idx.kspace_encode_step_1 == 1
    assert skeleton.acquisitions[0].idx.repetition == 0

    encoding = skeleton.header.encoding[0]
    assert encoding.trajectory.value == 'other'
    assert encoding.encodedSpace.matrixSize.x == 8
    assert encoding.encodedSpace.matrixSize.y == 2
    assert encoding.reconSpace.matrixSize.x == 4
    assert encoding.encodedSpace.fieldOfView_mm.x == 200.0
    assert skeleton.header.sequenceParameters.TR == [10.0]
    assert skeleton.header.sequenceParameters.TE == [2.0]
    assert skeleton.header.sequenceParameters.TI == [100.0]


def test_mrdskeleton_from_seq_uses_explicit_definitions(tmp_path: Path) -> None:
    """Explicit seq definitions should override the heuristic geometry defaults."""
    sequence_path = build_test_sequence_file(
        tmp_path / 'sequence.seq',
        encoding_matrix=(16, 4, 2),
        recon_matrix=(8, 4, 2),
        recon_fov=(0.18, 0.18, 0.01),
        trajectory_type='spiral',
    )

    skeleton = MRDSkeleton.from_seq(sequence_path)
    encoding = skeleton.header.encoding[0]

    assert encoding.trajectory.value == 'spiral'
    assert encoding.encodedSpace.matrixSize.x == 16
    assert encoding.encodedSpace.matrixSize.y == 4
    assert encoding.encodedSpace.matrixSize.z == 2
    assert encoding.reconSpace.matrixSize.x == 8
    assert encoding.reconSpace.fieldOfView_mm.x == 180.0


def test_mrdskeleton_from_seq_warns_and_uses_fallbacks(tmp_path: Path) -> None:
    """Missing optional geometry definitions should warn and fall back to heuristics."""
    sequence_path = build_test_sequence_file(tmp_path / 'sequence.seq', readout_oversampling_factor=2.0)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        skeleton = MRDSkeleton.from_seq(sequence_path)

    warning_messages = [str(warning.message) for warning in caught_warnings]

    assert 'Missing seq definition ReconFOV. Falling back to FOV.' in warning_messages
    assert 'Missing seq definition EncodingMatrix. Falling back to ADC and label heuristics.' in warning_messages
    assert 'Missing seq definition ReconMatrix. Falling back to encoding-matrix heuristics.' in warning_messages
    assert skeleton.header.encoding[0].reconSpace.fieldOfView_mm.x == 200.0


def test_mrdskeleton_from_seq_uses_adc_frequency_fallback_without_labels(tmp_path: Path) -> None:
    """Missing ADC labels should fall back to frequency classes and per-class LIN counters."""
    sequence_path = build_test_sequence_file(
        tmp_path / 'sequence.seq',
        include_labels=False,
        frequency_offsets=(10.0, 10.0),
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        skeleton = MRDSkeleton.from_seq(sequence_path)

    warning_messages = [str(warning.message) for warning in caught_warnings]

    assert (
        'No pulseq ADC labels found. LIN will increase within ADC frequency classes and class IDs will be stored in user_int[7].'
    ) in warning_messages
    assert skeleton.acquisitions[0].idx.kspace_encode_step_1 == 0
    assert skeleton.acquisitions[1].idx.kspace_encode_step_1 == 1
    assert skeleton.acquisitions[0].user_int[7] == 0
    assert skeleton.acquisitions[1].user_int[7] == 0
