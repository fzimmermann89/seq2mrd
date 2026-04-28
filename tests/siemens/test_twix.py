"""Tests for Siemens Twix metadata reading."""

from __future__ import annotations

from pathlib import Path

import ismrmrd
import numpy as np
import pytest

from seq2mrd.raw.siemens import SiemensRaw
from seq2mrd.raw.siemens.twix import SiemensTwixReader, TwixFormat
from seq2mrd.skeleton import MRDSkeleton
from tests.siemens._twix_data import SAMPLE_MEAS_HEADER, build_vb_file, build_vd_file, build_vd_multi_measurement_file


def test_header_reads_measurement_metadata_fields_from_vb_file(tmp_path: Path) -> None:
    """VB files should expose measurement-level fields through the parsed Siemens header."""
    file_path = tmp_path / 'vb.dat'
    file_path.write_bytes(build_vb_file())

    reader = SiemensTwixReader(file_path)
    header = reader.header()

    assert reader.format is TwixFormat.VB
    assert header.get_str('Meas.DICOM.tPatientName') == 'Ada Lovelace'
    assert header.get_str('Meas.DICOM.Manufacturer') == 'Siemens'
    assert header.get_str('Meas.DICOM.ManufacturersModelName') == 'Prisma'
    assert header.get_str('Meas.DICOM.StudyDate') == '20260427'
    assert header.get_str('Meas.DICOM.SeriesTime') == '123500.000'


def test_acquisitions_expose_per_acquisition_runtime_metadata(tmp_path: Path) -> None:
    """VD files should expose per-acquisition runtime metadata through acquisitions()."""
    file_path = tmp_path / 'vd.dat'
    file_path.write_bytes(build_vd_file())

    acquisition = next(SiemensTwixReader(file_path).acquisitions())
    metadata = acquisition.metadata

    assert metadata.measurement_uid == 19
    assert metadata.scan_counter == 4
    assert metadata.acquisition_time_stamp == 123456
    assert metadata.position == (1.5, 2.5, 3.5)
    assert metadata.read_dir == (0.0, 1.0, 0.0)
    assert metadata.phase_dir == (1.0, 0.0, 0.0)
    assert metadata.slice_dir == (0.0, 0.0, 1.0)
    assert metadata.patient_table_position == (10.0, 20.0, 30.0)


def test_acquisitions_read_complex_sample_data(tmp_path: Path) -> None:
    """The Twix reader should expose acquisition-aligned complex sample arrays."""
    file_path = tmp_path / 'vd.dat'
    file_path.write_bytes(build_vd_file())

    acquisition = next(SiemensTwixReader(file_path).acquisitions())

    assert acquisition.data.shape == (1, 2)
    assert acquisition.data[0, 0] == 1 + 2j
    assert acquisition.data[0, 1] == 3 + 4j


def test_vd_reader_uses_last_measurement_by_default(tmp_path: Path) -> None:
    """VD files should default to the last measurement."""
    file_path = tmp_path / 'vd_multi.dat'
    file_path.write_bytes(
        build_vd_multi_measurement_file(
            first_header=SAMPLE_MEAS_HEADER.replace('Ada Lovelace', 'First Measurement').replace('Prisma', 'Skyra'),
            second_header=SAMPLE_MEAS_HEADER,
        ),
    )

    reader = SiemensTwixReader(file_path)
    header = reader.header()
    acquisition = next(reader.acquisitions())

    assert header.get_str('Meas.DICOM.tPatientName') == 'Ada Lovelace'
    assert header.get_str('Meas.DICOM.ManufacturersModelName') == 'Prisma'
    assert acquisition.metadata.measurement_uid == 19
    assert acquisition.metadata.scan_counter == 4


def test_acquisitions_apply_scanner_discard_pre_and_post(tmp_path: Path) -> None:
    """The Twix reader should crop channels according to the scanner-side cutoff values."""
    file_path = tmp_path / 'vd_cutoff.dat'
    file_path.write_bytes(build_vd_file(discard_pre=1, discard_post=1, samples=((9 + 1j), (1 + 2j), (3 + 4j), (8 + 2j))))

    acquisition = next(SiemensTwixReader(file_path).acquisitions())

    assert acquisition.metadata.samples_in_scan == 2
    assert acquisition.metadata.discard_pre == 1
    assert acquisition.metadata.discard_post == 1
    assert acquisition.data.shape == (1, 2)
    assert acquisition.data[0, 0] == 1 + 2j
    assert acquisition.data[0, 1] == 3 + 4j


def test_siemens_raw_fills_skeleton_metadata_and_data(tmp_path: Path) -> None:
    """SiemensRaw should fill acquisition metadata and copy complex data into the skeleton."""
    file_path = tmp_path / 'vd.dat'
    file_path.write_bytes(build_vd_file())

    acquisition = ismrmrd.Acquisition()
    acquisition.resize(number_of_samples=2, active_channels=1, trajectory_dimensions=3)
    acquisition.traj[:] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    skeleton = MRDSkeleton(header=ismrmrd.xsd.ismrmrdHeader(), acquisitions=[acquisition])

    filled_skeleton = SiemensRaw(file_path)(skeleton)
    filled_acquisition = filled_skeleton.acquisitions[0]

    assert filled_skeleton.header.subjectInformation.patientName == 'Ada Lovelace'
    assert filled_skeleton.header.acquisitionSystemInformation.systemVendor == 'Siemens'
    assert filled_skeleton.header.acquisitionSystemInformation.systemModel == 'Prisma'
    assert filled_skeleton.header.studyInformation.studyDate == '2026-04-27'
    assert filled_skeleton.header.measurementInformation.seriesTime == '12:35:00.000'
    assert filled_acquisition.active_channels == 1
    assert filled_acquisition.data.shape == (1, 2)
    assert filled_acquisition.data[0, 0] == 1 + 2j
    assert filled_acquisition.data[0, 1] == 3 + 4j
    assert np.allclose(filled_acquisition.traj, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    assert filled_acquisition.acquisition_time_stamp == 123456
    assert tuple(filled_acquisition.position) == (1.5, 2.5, 3.5)


def test_siemens_raw_errors_for_extra_non_sync_acquisitions(tmp_path: Path) -> None:
    """Extra non-sync Twix acquisitions beyond the pulseq ADC count should fail."""
    file_path = tmp_path / 'vd_extra.dat'
    file_path.write_bytes(build_vd_file(record_count=2))

    acquisition = ismrmrd.Acquisition()
    acquisition.resize(number_of_samples=2, active_channels=1, trajectory_dimensions=3)
    acquisition.traj[:] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    skeleton = MRDSkeleton(header=ismrmrd.xsd.ismrmrdHeader(), acquisitions=[acquisition])

    with pytest.raises(
        ValueError,
        match='extra non-sync acquisitions that are not defined by the pulseq ADC events',
    ):
        SiemensRaw(file_path)(skeleton)
