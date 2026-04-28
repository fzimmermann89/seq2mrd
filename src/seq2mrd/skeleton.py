"""Pulseq-derived ISMRMRD skeleton creation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq


@dataclass(slots=True)
class Limits:
    """Encoding limit helper."""

    minimum: int = 0
    maximum: int = 0
    center: int = 0

    @classmethod
    def from_values(cls, values: np.ndarray | list[int]) -> Limits:
        """Create limits from an index list.

        Parameters
        ----------
        values
            Integer index values.

        Returns
        -------
            Limit description.
        """
        value_array = np.asarray(values, dtype=np.int64).reshape(-1)
        if value_array.size == 0:
            return cls()
        minimum = int(np.min(value_array))
        maximum = int(np.max(value_array))
        center = (maximum - minimum + 1) // 2
        return cls(minimum=minimum, maximum=maximum, center=center)

    def to_ismrmrd(self) -> ismrmrd.xsd.limitType:
        """Convert to an ISMRMRD limit type."""
        return ismrmrd.xsd.limitType(self.minimum, self.maximum, self.center)


def create_encoding_limits(adc_labels: dict[str, np.ndarray]) -> ismrmrd.xsd.encodingLimitsType:
    """Create encoding limits from pulseq labels."""
    return ismrmrd.xsd.encodingLimitsType(
        kspace_encoding_step_1=Limits.from_values(adc_labels.get('LIN', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        kspace_encoding_step_2=Limits.from_values(adc_labels.get('PAR', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        average=Limits.from_values(adc_labels.get('AVG', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        slice=Limits.from_values(adc_labels.get('SLC', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        contrast=Limits.from_values(adc_labels.get('ECO', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        phase=Limits.from_values(adc_labels.get('PHS', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        repetition=Limits.from_values(adc_labels.get('REP', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        set=Limits.from_values(adc_labels.get('SET', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
        segment=Limits.from_values(adc_labels.get('SEG', np.zeros(1, dtype=np.int64))).to_ismrmrd(),
    )


def build_header(
    *,
    field_of_view: tuple[float, float, float],
    recon_field_of_view: tuple[float, float, float],
    encoding_matrix: tuple[int, int, int],
    recon_matrix: tuple[int, int, int],
    trajectory_type: str,
    adc_labels: dict[str, np.ndarray],
    dwell_time_seconds: float,
    tr_ms: list[float],
    te_ms: list[float],
    ti_ms: list[float],
) -> ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader:
    """Build an ISMRMRD header from resolved pulseq values."""
    header = ismrmrd.xsd.ismrmrdHeader()
    user_parameters = ismrmrd.xsd.userParametersType()
    dwell_time_parameter = ismrmrd.xsd.userParameterDoubleType()
    dwell_time_parameter.name = 'dwellTime_us'
    dwell_time_parameter.value_ = dwell_time_seconds * 1e6
    user_parameters.userParameterDouble.append(dwell_time_parameter)
    header.userParameters = user_parameters

    sequence_parameters = ismrmrd.xsd.sequenceParametersType()
    sequence_parameters.TR = tr_ms
    sequence_parameters.TE = te_ms
    sequence_parameters.TI = ti_ms
    header.sequenceParameters = sequence_parameters

    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType(trajectory_type)

    encoded_space = ismrmrd.xsd.encodingSpaceType()
    encoded_space.matrixSize = ismrmrd.xsd.matrixSizeType(
        int(encoding_matrix[0]),
        int(encoding_matrix[1]),
        int(encoding_matrix[2]),
    )
    encoded_space.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(
        field_of_view[0] * 1e3,
        field_of_view[1] * 1e3,
        field_of_view[2] * 1e3,
    )

    recon_space = ismrmrd.xsd.encodingSpaceType()
    recon_space.matrixSize = ismrmrd.xsd.matrixSizeType(
        int(recon_matrix[0]),
        int(recon_matrix[1]),
        int(recon_matrix[2]),
    )
    recon_space.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(
        recon_field_of_view[0] * 1e3,
        recon_field_of_view[1] * 1e3,
        recon_field_of_view[2] * 1e3,
    )

    encoding.encodedSpace = encoded_space
    encoding.reconSpace = recon_space
    encoding.encodingLimits = create_encoding_limits(adc_labels)
    header.encoding.append(encoding)
    return header


def build_acquisition(
    *,
    number_of_samples: int,
    dwell_time_seconds: float,
    trajectory: np.ndarray,
    kspace_encode_step_1: int,
    kspace_encode_step_2: int,
    average: int,
    slice_: int,
    contrast: int,
    phase: int,
    repetition: int,
    set_: int,
    segment: int,
    user_int_7: int,
    is_phasecorr_data: bool,
    is_noise_measurement: bool,
) -> ismrmrd.Acquisition:
    """Build one ISMRMRD acquisition from resolved values."""
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(
        number_of_samples=number_of_samples,
        active_channels=1,
        trajectory_dimensions=trajectory.shape[1],
    )
    acquisition.data.fill(0)
    acquisition.traj[:] = trajectory
    acquisition.center_sample = int(np.argmin(np.abs(trajectory[:, 0])))
    acquisition.sample_time_us = float(dwell_time_seconds * 1e6)
    acquisition.idx.kspace_encode_step_1 = kspace_encode_step_1
    acquisition.idx.kspace_encode_step_2 = kspace_encode_step_2
    acquisition.idx.average = average
    acquisition.idx.slice = slice_
    acquisition.idx.contrast = contrast
    acquisition.idx.phase = phase
    acquisition.idx.repetition = repetition
    acquisition.idx.set = set_
    acquisition.idx.segment = segment
    acquisition.user_int[7] = user_int_7
    acquisition.read_dir = (1.0, 0.0, 0.0)
    acquisition.phase_dir = (0.0, 1.0, 0.0)
    acquisition.slice_dir = (0.0, 0.0, 1.0)
    if is_phasecorr_data:
        acquisition.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)
    if is_noise_measurement:
        acquisition.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
    return acquisition


@dataclass(slots=True)
class MRDSkeleton:
    """In-memory ISMRMRD skeleton created from pulseq."""

    header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader
    acquisitions: list[ismrmrd.Acquisition]

    @classmethod
    def from_seq(cls, seq_path: str | Path) -> MRDSkeleton:
        """Create an ISMRMRD skeleton from a pulseq file.

        Parameters
        ----------
        seq_path
            Path to the pulseq sequence file.

        Returns
        -------
            ISMRMRD skeleton with trajectory embedded into the acquisitions.
        """
        sequence = pypulseq.Sequence()
        sequence.read(str(seq_path))
        definitions = sequence.definitions

        adc_blocks = []
        for event_id in sequence.block_events:
            block = sequence.get_block(event_id)
            if block.adc is not None:
                adc_blocks.append(block.adc)
        if not adc_blocks:
            raise ValueError('The pulseq file does not contain ADC events.')

        number_of_acquisitions = len(adc_blocks)
        adc_labels = sequence.evaluate_labels(evolution='adc')
        has_labels = bool(adc_labels)
        zero_labels = np.zeros(number_of_acquisitions, dtype=np.int64)
        if has_labels:
            for label_name, label_values in adc_labels.items():
                label_array = np.asarray(label_values, dtype=np.int64).reshape(-1)
                if label_array.size == 0:
                    adc_labels[label_name] = zero_labels
                else:
                    adc_labels[label_name] = label_array
        else:
            warnings.warn(
                'No pulseq ADC labels found. LIN will increase within ADC frequency classes and class IDs will be stored in user_int[7].',
                stacklevel=2,
            )

        first_adc_block = adc_blocks[0]
        first_adc_number_of_samples = int(first_adc_block.num_samples)
        phase_encoding_limits = Limits.from_values(adc_labels.get('LIN', zero_labels))
        slice_encoding_limits = Limits.from_values(adc_labels.get('PAR', zero_labels))
        inferred_encoding_matrix = (
            first_adc_number_of_samples,
            max(phase_encoding_limits.maximum - phase_encoding_limits.minimum + 1, 1),
            max(slice_encoding_limits.maximum - slice_encoding_limits.minimum + 1, 1),
        )

        field_of_view_definition = definitions.get('FOV')
        field_of_view_values = np.asarray(field_of_view_definition, dtype=float).reshape(-1)
        if field_of_view_values.size != 3:
            raise ValueError('The pulseq FOV definition must contain exactly three values in meters.')
        field_of_view = (
            float(field_of_view_values[0]),
            float(field_of_view_values[1]),
            float(field_of_view_values[2]),
        )

        recon_field_of_view_definition = definitions.get('ReconFOV')
        if recon_field_of_view_definition is None:
            warnings.warn('Missing seq definition ReconFOV. Falling back to FOV.', stacklevel=2)
            recon_field_of_view = field_of_view
        else:
            recon_field_of_view_values = np.asarray(recon_field_of_view_definition, dtype=float).reshape(-1)
            if recon_field_of_view_values.size != 3:
                warnings.warn('Malformed seq definition ReconFOV. Falling back to FOV.', stacklevel=2)
                recon_field_of_view = field_of_view
            else:
                recon_field_of_view = (
                    float(recon_field_of_view_values[0]),
                    float(recon_field_of_view_values[1]),
                    float(recon_field_of_view_values[2]),
                )

        readout_oversampling_factor_definition = definitions.get('ReadoutOversamplingFactor', 1.0)
        readout_oversampling_factor = float(np.asarray(readout_oversampling_factor_definition, dtype=float).reshape(-1)[0])

        encoding_matrix_definition = definitions.get('EncodingMatrix')
        if encoding_matrix_definition is None:
            warnings.warn('Missing seq definition EncodingMatrix. Falling back to ADC and label heuristics.', stacklevel=2)
            encoding_matrix = inferred_encoding_matrix
        else:
            encoding_matrix_values = np.asarray(encoding_matrix_definition, dtype=int).reshape(-1)
            if encoding_matrix_values.size != 3:
                warnings.warn('Malformed seq definition EncodingMatrix. Falling back to ADC and label heuristics.', stacklevel=2)
                encoding_matrix = inferred_encoding_matrix
            else:
                encoding_matrix = (
                    int(encoding_matrix_values[0]),
                    int(encoding_matrix_values[1]),
                    int(encoding_matrix_values[2]),
                )

        inferred_recon_matrix = (
            round(encoding_matrix[0] / readout_oversampling_factor),
            encoding_matrix[1],
            encoding_matrix[2],
        )
        recon_matrix_definition = definitions.get('ReconMatrix')
        if recon_matrix_definition is None:
            warnings.warn('Missing seq definition ReconMatrix. Falling back to encoding-matrix heuristics.', stacklevel=2)
            recon_matrix = inferred_recon_matrix
        else:
            recon_matrix_values = np.asarray(recon_matrix_definition, dtype=int).reshape(-1)
            if recon_matrix_values.size != 3:
                warnings.warn('Malformed seq definition ReconMatrix. Falling back to encoding-matrix heuristics.', stacklevel=2)
                recon_matrix = inferred_recon_matrix
            else:
                recon_matrix = (
                    int(recon_matrix_values[0]),
                    int(recon_matrix_values[1]),
                    int(recon_matrix_values[2]),
                )

        trajectory_type_definition = definitions.get('TrajectoryType', 'other')
        if isinstance(trajectory_type_definition, str):
            trajectory_type = trajectory_type_definition.strip().lower()
        else:
            trajectory_type = str(trajectory_type_definition).strip().lower()
        if trajectory_type not in {'cartesian', 'epi', 'radial', 'spiral', 'other'}:
            warnings.warn('Malformed seq definition TrajectoryType. Falling back to "other".', stacklevel=2)
            trajectory_type = 'other'

        def normalize_sequence_parameter(definition_name: str) -> list[float]:
            definition_value = definitions.get(definition_name, [])
            if definition_value is None:
                return []
            if isinstance(definition_value, str):
                if not definition_value.strip():
                    return []
                return [float(definition_value) * 1e3]
            definition_array = np.asarray(definition_value, dtype=float).reshape(-1)
            return [float(value) * 1e3 for value in definition_array.tolist()]

        trajectory = np.asarray(sequence.calculate_kspace()[0], dtype=np.float32)
        if trajectory.shape[0] != 3:
            raise ValueError('Expected a three-dimensional pulseq k-space trajectory.')

        scaled_trajectory = trajectory.copy()
        for axis_index, axis_scale in enumerate(encoding_matrix):
            maximum_absolute_range = float(2 * np.max(np.abs(scaled_trajectory[axis_index])))
            if axis_scale < 2 or maximum_absolute_range < 1e-6:
                scaled_trajectory[axis_index] = 0.0
            else:
                scaled_trajectory[axis_index] = scaled_trajectory[axis_index] * (axis_scale / maximum_absolute_range)
        nonzero_axes = [
            axis_index for axis_index in range(scaled_trajectory.shape[0]) if float(np.max(np.abs(scaled_trajectory[axis_index]))) >= 1e-6
        ]
        if not nonzero_axes:
            raise ValueError('The pulseq trajectory does not contain any nonzero axes.')
        scaled_trajectory = scaled_trajectory[nonzero_axes]

        trajectory_arrays: list[np.ndarray] = []
        sample_offset = 0
        for adc_block in adc_blocks:
            current_number_of_samples = int(adc_block.num_samples)
            trajectory_arrays.append(scaled_trajectory[:, sample_offset : sample_offset + current_number_of_samples].T)
            sample_offset += current_number_of_samples
        if sample_offset != scaled_trajectory.shape[1]:
            raise ValueError('Pulseq trajectory samples do not match the ADC block sample counts.')

        header = build_header(
            field_of_view=field_of_view,
            recon_field_of_view=recon_field_of_view,
            encoding_matrix=encoding_matrix,
            recon_matrix=recon_matrix,
            trajectory_type=trajectory_type,
            adc_labels=adc_labels,
            dwell_time_seconds=float(first_adc_block.dwell),
            tr_ms=normalize_sequence_parameter('TR'),
            te_ms=normalize_sequence_parameter('TE'),
            ti_ms=normalize_sequence_parameter('TI'),
        )

        frequency_class_by_offset: dict[float, int] = {}
        line_index_by_frequency_offset: dict[float, int] = {}
        acquisitions = []
        for acquisition_index, (adc_block, trajectory_array) in enumerate(zip(adc_blocks, trajectory_arrays, strict=True)):
            user_int_7 = 0
            line_index = int(adc_labels.get('LIN', zero_labels)[acquisition_index])
            if not has_labels:
                frequency_offset = float(getattr(adc_block, 'freq_offset', 0.0))
                if frequency_offset not in frequency_class_by_offset:
                    frequency_class_by_offset[frequency_offset] = len(frequency_class_by_offset)
                    line_index_by_frequency_offset[frequency_offset] = 0
                user_int_7 = frequency_class_by_offset[frequency_offset]
                line_index = line_index_by_frequency_offset[frequency_offset]
                line_index_by_frequency_offset[frequency_offset] += 1

            acquisition = build_acquisition(
                number_of_samples=int(adc_block.num_samples),
                dwell_time_seconds=float(adc_block.dwell),
                trajectory=trajectory_array,
                kspace_encode_step_1=line_index,
                kspace_encode_step_2=int(adc_labels.get('PAR', zero_labels)[acquisition_index]),
                average=int(adc_labels.get('AVG', zero_labels)[acquisition_index]),
                slice_=int(adc_labels.get('SLC', zero_labels)[acquisition_index]),
                contrast=int(adc_labels.get('ECO', zero_labels)[acquisition_index]),
                phase=int(adc_labels.get('PHS', zero_labels)[acquisition_index]),
                repetition=int(adc_labels.get('REP', zero_labels)[acquisition_index]),
                set_=int(adc_labels.get('SET', zero_labels)[acquisition_index]),
                segment=int(adc_labels.get('SEG', zero_labels)[acquisition_index]),
                user_int_7=user_int_7,
                is_phasecorr_data=bool(adc_labels.get('NAV', zero_labels)[acquisition_index]),
                is_noise_measurement=bool(adc_labels.get('NOISE', zero_labels)[acquisition_index]),
            )
            acquisitions.append(acquisition)

        return cls(header=header, acquisitions=acquisitions)

    def write(self, path: str | Path, dataset_name: str = 'dataset') -> None:
        """Serialize the skeleton to an ISMRMRD file.

        Parameters
        ----------
        path
            Output file path.
        dataset_name
            Name of the dataset inside the HDF5 file.
        """
        with ismrmrd.Dataset(str(path), dataset_name=dataset_name, create_if_needed=True) as dataset:
            dataset.write_xml_header(self.header.toXML())
            for acquisition in self.acquisitions:
                dataset.append_acquisition(acquisition)
