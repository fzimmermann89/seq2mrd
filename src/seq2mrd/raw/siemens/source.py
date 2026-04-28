"""Siemens raw data source."""

from pathlib import Path

import ismrmrd

from seq2mrd.raw.base import RawSource
from seq2mrd.raw.siemens.twix import SiemensTwixReader, first_present, normalize_siemens_date, normalize_siemens_time
from seq2mrd.skeleton import MRDSkeleton


class SiemensRaw(RawSource):
    """Siemens raw data enricher."""

    def __init__(self, raw_path: str | Path) -> None:
        """Create a Siemens raw data source.

        Parameters
        ----------
        raw_path
            Path to the Siemens Twix file.
        """
        super().__init__(raw_path)
        self.twix_reader = SiemensTwixReader(raw_path)

    def __call__(self, skeleton: MRDSkeleton) -> MRDSkeleton:
        """Fill a pulseq-derived skeleton with Siemens raw data.

        Parameters
        ----------
        skeleton
            Skeleton created from the pulseq file.

        Returns
        -------
            Updated skeleton.
        """
        header = self.twix_reader.header()
        readout_oversampling_factor = header.get_float('DICOM.flReadoutOSFactor')
        remove_oversampling = header.get_str('MEAS.sSpecPara.ucRemoveOversampling')

        patient_name = first_present(
            header.get_str('DICOM.tPatientName'),
            header.get_str('HEADER.tPatientName'),
        )
        vendor = first_present(
            header.get_str('DICOM.Manufacturer'),
            'Siemens',
        )
        system_model = header.get_str('DICOM.ManufacturersModelName')
        study_date = normalize_siemens_date(
            first_present(
                header.get_str('DICOM.StudyDate'),
                header.get_str('HEADER.StudyDate'),
            ),
        )
        study_time = normalize_siemens_time(
            first_present(
                header.get_str('DICOM.StudyTime'),
                header.get_str('HEADER.StudyTime'),
            ),
        )
        series_date = normalize_siemens_date(
            first_present(
                header.get_str('DICOM.SeriesDate'),
                header.get_str('HEADER.SeriesDate'),
            ),
        )
        series_time = normalize_siemens_time(
            first_present(
                header.get_str('DICOM.SeriesTime'),
                header.get_str('HEADER.SeriesTime'),
            ),
        )

        if patient_name:
            subject_information = skeleton.header.subjectInformation or ismrmrd.xsd.subjectInformationType()
            subject_information.patientName = patient_name
            skeleton.header.subjectInformation = subject_information

        acquisition_system_information = skeleton.header.acquisitionSystemInformation or ismrmrd.xsd.acquisitionSystemInformationType()
        if vendor:
            acquisition_system_information.systemVendor = vendor
        if system_model:
            acquisition_system_information.systemModel = system_model
        skeleton.header.acquisitionSystemInformation = acquisition_system_information

        if study_date or study_time:
            study_information = skeleton.header.studyInformation or ismrmrd.xsd.studyInformationType()
            if study_date:
                study_information.studyDate = study_date
            if study_time:
                study_information.studyTime = study_time
            skeleton.header.studyInformation = study_information

        if series_date or series_time:
            measurement_information = skeleton.header.measurementInformation or ismrmrd.xsd.measurementInformationType()
            if series_date:
                measurement_information.seriesDate = series_date
            if series_time:
                measurement_information.seriesTime = series_time
            skeleton.header.measurementInformation = measurement_information

        twix_acquisition_iterator = self.twix_reader.acquisitions()
        for acquisition in skeleton.acquisitions:
            try:
                twix_acquisition = next(twix_acquisition_iterator)
            except StopIteration as exception:
                raise ValueError(
                    'The Siemens Twix file contains fewer non-sync acquisitions than the pulseq file defines ADC events.',
                ) from exception

            if twix_acquisition.data.shape[1] != acquisition.number_of_samples:
                raise ValueError(
                    'The Siemens sample count does not match the pulseq ADC sample count. '
                    f'Got {twix_acquisition.data.shape[1]} Siemens samples and {acquisition.number_of_samples} pulseq samples. '
                    f'Siemens header hints: flReadoutOSFactor={readout_oversampling_factor!r}, '
                    f'ucRemoveOversampling={remove_oversampling!r}.',
                )

            trajectory = acquisition.traj.copy()
            trajectory_dimensions = acquisition.trajectory_dimensions
            acquisition.resize(
                number_of_samples=twix_acquisition.data.shape[1],
                active_channels=twix_acquisition.data.shape[0],
                trajectory_dimensions=trajectory_dimensions,
            )
            acquisition.traj[:] = trajectory
            acquisition.data[:] = twix_acquisition.data

            metadata = twix_acquisition.metadata
            acquisition.measurement_uid = metadata.measurement_uid
            acquisition.scan_counter = metadata.scan_counter
            acquisition.acquisition_time_stamp = metadata.acquisition_time_stamp
            acquisition.physiology_time_stamp[:] = (metadata.physiology_time_stamp, 0, 0)
            acquisition.available_channels = metadata.used_channels
            acquisition.discard_pre = metadata.discard_pre
            acquisition.discard_post = metadata.discard_post
            acquisition.position[:] = metadata.position
            acquisition.read_dir[:] = metadata.read_dir
            acquisition.phase_dir[:] = metadata.phase_dir
            acquisition.slice_dir[:] = metadata.slice_dir
            acquisition.patient_table_position[:] = metadata.patient_table_position

        try:
            next(twix_acquisition_iterator)
        except StopIteration:
            return skeleton

        raise ValueError(
            'The Siemens Twix file contains extra non-sync acquisitions that are not defined by the pulseq ADC events.',
        )
