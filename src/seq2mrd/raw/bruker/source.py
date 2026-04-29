"""Bruker raw data source."""

from __future__ import annotations

import datetime as dt

import ismrmrd
import numpy as np

from seq2mrd.raw.base import RawSource
from seq2mrd.raw.bruker.fid import decode_fid
from seq2mrd.raw.bruker.parameters import first_array, first_float, first_int, first_str, load_parameter_files
from seq2mrd.skeleton import MRDSkeleton


def parse_bruker_datetime(value: str | None) -> dt.datetime | None:
    """Parse a Bruker acquisition time string."""
    if not value:
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%S,%f%z', '%H:%M:%S %d %b %Y'):
        try:
            timestamp = dt.datetime.strptime(value.strip('<>'), fmt)
        except ValueError:
            continue
        if timestamp.tzinfo is not None:
            return timestamp.replace(tzinfo=None)
        return timestamp
    return None


class BrukerRaw(RawSource):
    """Bruker raw data enricher."""

    def __call__(self, skeleton: MRDSkeleton) -> MRDSkeleton:
        """Fill a pulseq-derived skeleton with Bruker raw data.

        Parameters
        ----------
        skeleton
            Skeleton created from the pulseq file.

        Returns
        -------
            Updated skeleton.
        """
        parameter_files = load_parameter_files(self.raw_path)
        data = decode_fid(self.raw_path, parameter_files)

        if data.shape[0] != len(skeleton.acquisitions):
            raise ValueError(
                'The Bruker raw data contains a different number of ADC readouts than the pulseq skeleton. '
                f'Got {data.shape[0]} readouts and {len(skeleton.acquisitions)} acquisitions.',
            )

        patient_name = first_str(parameter_files, 'SUBJECT_name_string', 'VisuSubjectName', 'SUBJECT_id')
        vendor = 'Bruker'
        acquisition_system_information = skeleton.header.acquisitionSystemInformation or ismrmrd.xsd.acquisitionSystemInformationType()
        acquisition_system_information.systemVendor = vendor
        skeleton.header.acquisitionSystemInformation = acquisition_system_information

        if patient_name:
            subject_information = skeleton.header.subjectInformation or ismrmrd.xsd.subjectInformationType()
            subject_information.patientName = patient_name
            skeleton.header.subjectInformation = subject_information

        acquisition_time = parse_bruker_datetime(first_str(parameter_files, 'ACQ_time'))
        subject_position = first_str(parameter_files, 'SUBJECT_position')
        if acquisition_time is not None or subject_position:
            measurement_information = skeleton.header.measurementInformation or ismrmrd.xsd.measurementInformationType()
            if acquisition_time is not None:
                measurement_information.seriesDate = acquisition_time.date().isoformat()
                measurement_information.seriesTime = acquisition_time.time().isoformat()
            if subject_position:
                measurement_information.patientPosition = subject_position
            skeleton.header.measurementInformation = measurement_information

        larmor_frequency_mhz = first_float(parameter_files, 'PVM_FrqWork', 'PVM_FrqRef', 'VisuAcqImagingFrequency')
        if larmor_frequency_mhz is not None:
            experimental_conditions = skeleton.header.experimentalConditions or ismrmrd.xsd.experimentalConditionsType()
            experimental_conditions.H1resonanceFrequency_Hz = int(larmor_frequency_mhz * 1e6)
            skeleton.header.experimentalConditions = experimental_conditions

        position_matrices = {
            'Head_Supine': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
            'Head_Prone': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
            'Head_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            'Head_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
            'Foot_Supine': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
            'Tail_Supine': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
            'Foot_Prone': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
            'Tail_Prone': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
            'Foot_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            'Tail_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            'Foot_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
            'Tail_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
        }
        subject_rotation = position_matrices.get(subject_position or '', np.eye(3, dtype=np.float32))

        acquisition_count = len(skeleton.acquisitions)

        def select_frames(values: np.ndarray, width: int) -> np.ndarray | None:
            reshaped_values = values.reshape(-1, width)
            if reshaped_values.shape[0] == 1:
                return np.repeat(reshaped_values, acquisition_count, axis=0)
            if reshaped_values.shape[0] == acquisition_count:
                return reshaped_values
            return None

        read_direction = None
        phase_direction = None
        slice_direction = None
        if (
            visu_orientation := first_array(parameter_files, 'VisuCoreOrientation', dtype=np.float32)
        ) is not None and visu_orientation.size:
            selected_orientation = select_frames(visu_orientation, 9)
            if selected_orientation is not None:
                orientation_matrices = subject_rotation @ np.transpose(selected_orientation.reshape(-1, 3, 3), (0, 2, 1))
                read_direction = orientation_matrices[:, :, 0]
                phase_direction = orientation_matrices[:, :, 1]
                slice_direction = orientation_matrices[:, :, 2]
        elif (
            grad_orientation := first_array(parameter_files, 'PVM_SPackArrGradOrient', dtype=np.float32)
        ) is not None and grad_orientation.size:
            selected_orientation = select_frames(grad_orientation, 9)
            if selected_orientation is not None:
                orientation_matrices = subject_rotation @ selected_orientation.reshape(-1, 3, 3)
                read_direction = orientation_matrices[:, :, 0]
                phase_direction = orientation_matrices[:, :, 1]
                slice_direction = orientation_matrices[:, :, 2]

        position = None
        if (visu_position := first_array(parameter_files, 'VisuCorePosition', dtype=np.float32)) is not None and visu_position.size:
            selected_position = select_frames(visu_position, 3)
            if selected_position is not None:
                position = selected_position
                if (
                    read_direction is not None
                    and phase_direction is not None
                    and slice_direction is not None
                    and (visu_extent := first_array(parameter_files, 'VisuCoreExtent', dtype=np.float32)) is not None
                    and visu_extent.size >= 2
                ):
                    extent_xyz = np.zeros(3, dtype=np.float32)
                    extent_xyz[: min(3, visu_extent.size)] = visu_extent.reshape(-1)[:3]
                    if first_int(parameter_files, 'VisuCoreDim') != 3:
                        extent_xyz[2] = 0
                    position_shift = extent_xyz[0] * read_direction / 2
                    position_shift = position_shift + extent_xyz[1] * phase_direction / 2
                    position_shift = position_shift + extent_xyz[2] * slice_direction / 2
                    position = position + position_shift

        for acquisition, readout_index in zip(skeleton.acquisitions, range(data.shape[0]), strict=True):
            readout = data[readout_index]
            if readout.shape[1] != acquisition.number_of_samples:
                raise ValueError(
                    'The Bruker readout length does not match the pulseq ADC sample count. '
                    f'Got {readout.shape[1]} samples and {acquisition.number_of_samples} pulseq samples.',
                )

            trajectory = acquisition.traj.copy()
            trajectory_dimensions = acquisition.trajectory_dimensions
            acquisition.resize(
                number_of_samples=readout.shape[1],
                active_channels=readout.shape[0],
                trajectory_dimensions=trajectory_dimensions,
            )
            acquisition.traj[:] = trajectory
            acquisition.data[:] = readout
            if position is not None:
                acquisition.position[:] = tuple(float(value) for value in position[readout_index])
            if read_direction is not None:
                acquisition.read_dir[:] = tuple(float(value) for value in read_direction[readout_index])
            if phase_direction is not None:
                acquisition.phase_dir[:] = tuple(float(value) for value in phase_direction[readout_index])
            if slice_direction is not None:
                acquisition.slice_dir[:] = tuple(float(value) for value in slice_direction[readout_index])

        return skeleton
