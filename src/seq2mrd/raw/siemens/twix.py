"""Minimal Siemens Twix readers for metadata extraction."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import numpy as np

from seq2mrd.raw.siemens.xprotocol import SiemensHeader, parse_siemens_header

RAID_FILE_HEADER_SIZE = 8
RAID_FILE_ENTRY_SIZE = 152
RAID_NAME_FIELD_SIZE = 64
MEASUREMENT_BUFFER_NAME_LIMIT = 32
MDH_DMA_LENGTH_MASK = 0x01FF_FFFF
EVAL_INFO_LAST_SCAN_IN_MEASUREMENT = 0
EVAL_INFO_SYNCDATA = 5
VB_MDH_SIZE = 128
VD_SCAN_HEADER_SIZE = 192
CHANNEL_HEADER_SIZE = 32

VB_MDH_OFFSETS = {
    'flags_and_dma_length': 0,
    'measurement_uid': 4,
    'scan_counter': 8,
    'time_stamp': 12,
    'pmu_time_stamp': 16,
    'eval_info_mask': 20,
    'samples_in_scan': 28,
    'used_channels': 30,
    'discard_pre': 60,
    'discard_post': 62,
    'slice_data': 96,
    'channel_id': 124,
    'patient_table_position_z': 126,
}

VD_SCAN_HEADER_OFFSETS = {
    'flags_and_dma_length': 0,
    'measurement_uid': 4,
    'scan_counter': 8,
    'time_stamp': 12,
    'pmu_time_stamp': 16,
    'patient_table_position_x': 24,
    'patient_table_position_y': 28,
    'patient_table_position_z': 32,
    'eval_info_mask': 40,
    'samples_in_scan': 48,
    'used_channels': 50,
    'discard_pre': 80,
    'discard_post': 82,
    'slice_data': 100,
}

SLICE_DATA_OFFSETS = {
    'position_x': 0,
    'position_y': 4,
    'position_z': 8,
    'quaternion_w': 12,
    'quaternion_x': 16,
    'quaternion_y': 20,
    'quaternion_z': 24,
}


class TwixFormat(Enum):
    """Supported Siemens Twix container layouts."""

    VB = 'vb'
    VD = 'vd'


@dataclass(slots=True)
class MeasurementHeaderBuffer:
    """One Siemens measurement header buffer."""

    name: str
    data: str


@dataclass(slots=True)
class TwixMeasurement:
    """One Siemens measurement description."""

    number: int
    buffers: list[MeasurementHeaderBuffer]
    data_offset: int
    end_offset: int | None
    is_vb: bool
    patient_name: str | None = None
    protocol_name: str | None = None


@dataclass(slots=True)
class SiemensAcquisitionMetadata:
    """Runtime metadata extracted for one Siemens acquisition."""

    measurement_uid: int
    scan_counter: int
    acquisition_time_stamp: int
    physiology_time_stamp: int
    position: tuple[float, float, float]
    read_dir: tuple[float, float, float]
    phase_dir: tuple[float, float, float]
    slice_dir: tuple[float, float, float]
    patient_table_position: tuple[float, float, float]
    samples_in_scan: int
    used_channels: int
    discard_pre: int
    discard_post: int


@dataclass(slots=True)
class VdRaidEntry:
    """One Siemens VD RAID table entry."""

    measurement_number: int
    offset: int
    length: int
    patient_name: str
    protocol_name: str


@dataclass(slots=True)
class TwixScanHeader:
    """Minimal parsed scan-header information."""

    measurement_uid: int
    scan_counter: int
    acquisition_time_stamp: int
    physiology_time_stamp: int
    eval_info_mask: tuple[int, int]
    samples_in_scan: int
    used_channels: int
    discard_pre: int
    discard_post: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    patient_table_position: tuple[float, float, float]


@dataclass(slots=True)
class TwixAcquisition:
    """One Siemens acquisition with runtime metadata and complex samples."""

    metadata: SiemensAcquisitionMetadata
    data: np.ndarray


class SiemensTwixReader:
    """Read Siemens Twix metadata without sequence-specific conversion logic."""

    def __init__(self, path: str | Path) -> None:
        """Store the Siemens raw-data path and detect the Twix layout.

        Parameters
        ----------
        path
            Path to the Siemens Twix file.
        """
        self.path = Path(path)
        self.format = detect_twix_format(self.path)

    @property
    def is_vb(self) -> bool:
        """Return whether the Twix file uses the VB layout.

        Returns
        -------
            True for VB files.
        """
        return self.format is TwixFormat.VB

    def _read_measurement(self, measurement_number: int | None = None) -> TwixMeasurement:
        """Read one Twix measurement preamble.

        Parameters
        ----------
        measurement_number
            One-based measurement number. Defaults to the last measurement for VD files
            and the only measurement for VB files.

        Returns
        -------
            Parsed measurement description.
        """
        if self.is_vb:
            if measurement_number is None:
                measurement_number = 1
            if measurement_number != 1:
                raise ValueError('VB Twix files contain a single measurement.')
            with self.path.open('rb') as handle:
                _header_dma_length, buffers = read_measurement_preamble(handle, base_offset=0)
                return TwixMeasurement(
                    number=1,
                    buffers=buffers,
                    data_offset=handle.tell(),
                    end_offset=None,
                    is_vb=True,
                )

        raid_entries = read_vd_raid_entries(self.path)
        if measurement_number is None:
            measurement_number = len(raid_entries)
        if measurement_number < 1 or measurement_number > len(raid_entries):
            raise ValueError(
                f'Measurement {measurement_number} is out of range for a file with {len(raid_entries)} measurements.',
            )

        entry = raid_entries[measurement_number - 1]
        with self.path.open('rb') as handle:
            handle.seek(entry.offset)
            _header_dma_length, buffers = read_measurement_preamble(handle, base_offset=entry.offset)
            return TwixMeasurement(
                number=measurement_number,
                buffers=buffers,
                data_offset=handle.tell(),
                end_offset=entry.offset + entry.length,
                is_vb=False,
                patient_name=entry.patient_name or None,
                protocol_name=entry.protocol_name or None,
            )

    def header(self, measurement_number: int | None = None) -> SiemensHeader:
        """Parse the Siemens `Meas` header buffer.

        Parameters
        ----------
        measurement_number
            One-based measurement number. Defaults to the last measurement for VD files
            and the only measurement for VB files.

        Returns
        -------
            Parsed Siemens XProtocol header.
        """
        measurement = self._read_measurement(measurement_number)
        meas_buffer = next((buffer for buffer in measurement.buffers if buffer.name == 'Meas'), None)
        if meas_buffer is None:
            raise ValueError('No Meas buffer found in Siemens measurement header.')
        return parse_siemens_header(meas_buffer.data.rstrip('\x00'))

    def acquisitions(self, measurement_number: int | None = None) -> Iterator[TwixAcquisition]:
        """Return non-sync Siemens acquisitions with complex sample data.

        Parameters
        ----------
        measurement_number
            One-based measurement number. Defaults to the last measurement for VD files
            and the only measurement for VB files.

        Returns
        -------
            Iterator over parsed Siemens acquisitions.
        """
        measurement = self._read_measurement(measurement_number)
        with self.path.open('rb') as handle:
            handle.seek(measurement.data_offset)
            if measurement.is_vb:
                yield from iter_vb_acquisitions(handle)
                return
            if measurement.end_offset is None:
                raise ValueError('VD measurements require a finite end offset.')
            yield from iter_vd_acquisitions(handle, end_offset=measurement.end_offset)


def detect_twix_format(path: Path) -> TwixFormat:
    """Detect whether a Siemens Twix file uses the VB or VD layout.

    Parameters
    ----------
    path
        Path to the Siemens Twix file.

    Returns
    -------
        Detected Twix layout.
    """
    with path.open('rb') as handle:
        header = read_exact(handle, RAID_FILE_HEADER_SIZE)
    hd_size = int.from_bytes(header[:4], 'little')
    return TwixFormat.VD if hd_size == 0 else TwixFormat.VB


def read_vd_raid_entries(path: Path) -> list[VdRaidEntry]:
    """Read the Siemens VD RAID measurement table.

    Parameters
    ----------
    path
        Path to the Siemens Twix file.

    Returns
    -------
        Parsed RAID entries.
    """
    with path.open('rb') as handle:
        header = read_exact(handle, RAID_FILE_HEADER_SIZE)
        hd_size, count = struct.unpack('<II', header)
        if hd_size != 0:
            raise ValueError('Expected a VD RAID file with hdSize == 0.')
        entries: list[VdRaidEntry] = []
        for measurement_index in range(count):
            raw_entry = read_exact(handle, RAID_FILE_ENTRY_SIZE)
            _measurement_id, _file_id, offset, length = struct.unpack_from('<IIQQ', raw_entry, 0)
            patient_name = read_c_string_from_bytes(raw_entry[24 : 24 + RAID_NAME_FIELD_SIZE])
            protocol_name = read_c_string_from_bytes(raw_entry[88 : 88 + RAID_NAME_FIELD_SIZE])
            entries.append(
                VdRaidEntry(
                    measurement_number=measurement_index + 1,
                    offset=offset,
                    length=length,
                    patient_name=patient_name,
                    protocol_name=protocol_name,
                )
            )
        return entries


def read_measurement_preamble(
    handle: BinaryIO,
    *,
    base_offset: int,
) -> tuple[int, list[MeasurementHeaderBuffer]]:
    """Read a Siemens measurement header preamble.

    Parameters
    ----------
    handle
        Open file handle positioned at the measurement start.
    base_offset
        Absolute file offset for the current measurement.

    Returns
    -------
        DMA length and parsed header buffers.
    """
    header_dma_length = int.from_bytes(read_exact(handle, 4), 'little')
    if header_dma_length <= 0:
        raise ValueError('Invalid Siemens measurement header with non-positive DMA length.')
    buffer_count = int.from_bytes(read_exact(handle, 4), 'little')
    buffers = read_measurement_buffers(handle, buffer_count)
    align_to_32_bytes(handle, base_offset=base_offset)
    return header_dma_length, buffers


def read_measurement_buffers(handle: BinaryIO, buffer_count: int) -> list[MeasurementHeaderBuffer]:
    """Read Siemens measurement header buffers.

    Parameters
    ----------
    handle
        Open file handle positioned at the first measurement buffer.
    buffer_count
        Number of measurement buffers.

    Returns
    -------
        Parsed measurement header buffers.
    """
    buffers: list[MeasurementHeaderBuffer] = []
    for _ in range(buffer_count):
        name = read_c_string(handle)
        data_length = int.from_bytes(read_exact(handle, 4), 'little')
        raw_data = read_exact(handle, data_length)
        buffers.append(
            MeasurementHeaderBuffer(
                name=name,
                data=raw_data.decode('latin1', errors='ignore').rstrip('\x00'),
            )
        )
    return buffers


def iter_vb_acquisitions(handle: BinaryIO) -> Iterator[TwixAcquisition]:
    """Iterate over VB acquisitions with complex samples.

    Parameters
    ----------
    handle
        Open file handle positioned at the first VB record.

    Yields
    ------
        Parsed Siemens acquisitions.
    """
    while True:
        raw_header = handle.read(VB_MDH_SIZE)
        if len(raw_header) < VB_MDH_SIZE:
            return
        scan_header = parse_vb_scan_header(raw_header)
        record_bytes_remaining = max(dma_length(raw_header) - VB_MDH_SIZE, 0)
        samples_in_scan = scan_header.samples_in_scan
        used_channels = scan_header.used_channels
        is_sync_data = has_eval_info_bit(scan_header.eval_info_mask, EVAL_INFO_SYNCDATA)
        channels: list[np.ndarray] = []
        if not is_sync_data and used_channels > 0:
            channels.append(
                np.frombuffer(read_exact(handle, samples_in_scan * 8), dtype=np.complex64, count=samples_in_scan).copy(),
            )
            for _channel_index in range(1, used_channels):
                read_exact(handle, VB_MDH_SIZE)
                channels.append(
                    np.frombuffer(read_exact(handle, samples_in_scan * 8), dtype=np.complex64, count=samples_in_scan).copy(),
                )
        consumed_bytes = max(used_channels * (VB_MDH_SIZE + samples_in_scan * 8) - VB_MDH_SIZE, 0)
        if record_bytes_remaining > consumed_bytes:
            handle.seek(record_bytes_remaining - consumed_bytes, 1)
        if not is_sync_data:
            cropped_channels = crop_channels(channels, scan_header.discard_pre, scan_header.discard_post)
            yield TwixAcquisition(
                metadata=acquisition_metadata_from_scan_header(scan_header),
                data=np.stack(cropped_channels, axis=0).astype(np.complex64, copy=False),
            )
        if has_eval_info_bit(scan_header.eval_info_mask, EVAL_INFO_LAST_SCAN_IN_MEASUREMENT):
            return


def iter_vd_acquisitions(handle: BinaryIO, *, end_offset: int) -> Iterator[TwixAcquisition]:
    """Iterate over VD acquisitions with complex samples.

    Parameters
    ----------
    handle
        Open file handle positioned at the first VD record.
    end_offset
        Absolute end offset for the current measurement.

    Yields
    ------
        Parsed Siemens acquisitions.
    """
    while handle.tell() + VD_SCAN_HEADER_SIZE <= end_offset:
        raw_header = handle.read(VD_SCAN_HEADER_SIZE)
        if len(raw_header) < VD_SCAN_HEADER_SIZE:
            return
        scan_header = parse_vd_scan_header(raw_header)
        record_bytes_remaining = max(dma_length(raw_header) - VD_SCAN_HEADER_SIZE, 0)
        samples_in_scan = scan_header.samples_in_scan
        used_channels = scan_header.used_channels
        is_sync_data = has_eval_info_bit(scan_header.eval_info_mask, EVAL_INFO_SYNCDATA)
        if not is_sync_data and samples_in_scan == 0 and used_channels == 0:
            if record_bytes_remaining > 0:
                handle.seek(record_bytes_remaining, 1)
            if has_eval_info_bit(scan_header.eval_info_mask, EVAL_INFO_LAST_SCAN_IN_MEASUREMENT):
                return
            continue
        channels: list[np.ndarray] = []
        for _channel_index in range(used_channels):
            read_exact(handle, CHANNEL_HEADER_SIZE)
            channels.append(
                np.frombuffer(read_exact(handle, samples_in_scan * 8), dtype=np.complex64, count=samples_in_scan).copy(),
            )
        consumed_bytes = used_channels * (CHANNEL_HEADER_SIZE + samples_in_scan * 8)
        if record_bytes_remaining > consumed_bytes:
            handle.seek(record_bytes_remaining - consumed_bytes, 1)
        if not is_sync_data:
            cropped_channels = crop_channels(channels, scan_header.discard_pre, scan_header.discard_post)
            yield TwixAcquisition(
                metadata=acquisition_metadata_from_scan_header(scan_header),
                data=np.stack(cropped_channels, axis=0).astype(np.complex64, copy=False),
            )
        if has_eval_info_bit(scan_header.eval_info_mask, EVAL_INFO_LAST_SCAN_IN_MEASUREMENT):
            return


def parse_vb_scan_header(raw_header: bytes) -> TwixScanHeader:
    """Parse a Siemens VB MDH into common scan-header metadata.

    Parameters
    ----------
    raw_header
        Raw VB MDH bytes.

    Returns
    -------
        Parsed scan-header metadata.
    """
    eval_info_mask = struct.unpack_from('<II', raw_header, VB_MDH_OFFSETS['eval_info_mask'])
    return TwixScanHeader(
        measurement_uid=struct.unpack_from('<i', raw_header, VB_MDH_OFFSETS['measurement_uid'])[0],
        scan_counter=struct.unpack_from('<I', raw_header, VB_MDH_OFFSETS['scan_counter'])[0],
        acquisition_time_stamp=struct.unpack_from('<I', raw_header, VB_MDH_OFFSETS['time_stamp'])[0],
        physiology_time_stamp=struct.unpack_from('<I', raw_header, VB_MDH_OFFSETS['pmu_time_stamp'])[0],
        eval_info_mask=eval_info_mask,
        samples_in_scan=struct.unpack_from('<H', raw_header, VB_MDH_OFFSETS['samples_in_scan'])[0],
        used_channels=struct.unpack_from('<H', raw_header, VB_MDH_OFFSETS['used_channels'])[0],
        discard_pre=struct.unpack_from('<H', raw_header, VB_MDH_OFFSETS['discard_pre'])[0],
        discard_post=struct.unpack_from('<H', raw_header, VB_MDH_OFFSETS['discard_post'])[0],
        position=parse_slice_position(raw_header, VB_MDH_OFFSETS['slice_data']),
        quaternion=parse_slice_quaternion(raw_header, VB_MDH_OFFSETS['slice_data']),
        patient_table_position=(0.0, 0.0, float(struct.unpack_from('<H', raw_header, VB_MDH_OFFSETS['patient_table_position_z'])[0])),
    )


def parse_vd_scan_header(raw_header: bytes) -> TwixScanHeader:
    """Parse a Siemens VD scan header into common scan-header metadata.

    Parameters
    ----------
    raw_header
        Raw VD scan-header bytes.

    Returns
    -------
        Parsed scan-header metadata.
    """
    eval_info_mask = struct.unpack_from('<II', raw_header, VD_SCAN_HEADER_OFFSETS['eval_info_mask'])
    return TwixScanHeader(
        measurement_uid=struct.unpack_from('<i', raw_header, VD_SCAN_HEADER_OFFSETS['measurement_uid'])[0],
        scan_counter=struct.unpack_from('<I', raw_header, VD_SCAN_HEADER_OFFSETS['scan_counter'])[0],
        acquisition_time_stamp=struct.unpack_from('<I', raw_header, VD_SCAN_HEADER_OFFSETS['time_stamp'])[0],
        physiology_time_stamp=struct.unpack_from('<I', raw_header, VD_SCAN_HEADER_OFFSETS['pmu_time_stamp'])[0],
        eval_info_mask=eval_info_mask,
        samples_in_scan=struct.unpack_from('<H', raw_header, VD_SCAN_HEADER_OFFSETS['samples_in_scan'])[0],
        used_channels=struct.unpack_from('<H', raw_header, VD_SCAN_HEADER_OFFSETS['used_channels'])[0],
        discard_pre=struct.unpack_from('<H', raw_header, VD_SCAN_HEADER_OFFSETS['discard_pre'])[0],
        discard_post=struct.unpack_from('<H', raw_header, VD_SCAN_HEADER_OFFSETS['discard_post'])[0],
        position=parse_slice_position(raw_header, VD_SCAN_HEADER_OFFSETS['slice_data']),
        quaternion=parse_slice_quaternion(raw_header, VD_SCAN_HEADER_OFFSETS['slice_data']),
        patient_table_position=(
            float(struct.unpack_from('<i', raw_header, VD_SCAN_HEADER_OFFSETS['patient_table_position_x'])[0]),
            float(struct.unpack_from('<i', raw_header, VD_SCAN_HEADER_OFFSETS['patient_table_position_y'])[0]),
            float(struct.unpack_from('<i', raw_header, VD_SCAN_HEADER_OFFSETS['patient_table_position_z'])[0]),
        ),
    )


def parse_slice_position(raw_header: bytes, slice_offset: int) -> tuple[float, float, float]:
    """Parse the Siemens slice position vector.

    Parameters
    ----------
    raw_header
        Raw scan-header bytes.
    slice_offset
        Offset to the Siemens slice-data block.

    Returns
    -------
        Slice position vector.
    """
    return (
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['position_x'])[0],
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['position_y'])[0],
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['position_z'])[0],
    )


def parse_slice_quaternion(raw_header: bytes, slice_offset: int) -> tuple[float, float, float, float]:
    """Parse the Siemens slice quaternion.

    Parameters
    ----------
    raw_header
        Raw scan-header bytes.
    slice_offset
        Offset to the Siemens slice-data block.

    Returns
    -------
        Siemens quaternion in stored order.
    """
    return (
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['quaternion_w'])[0],
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['quaternion_x'])[0],
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['quaternion_y'])[0],
        struct.unpack_from('<f', raw_header, slice_offset + SLICE_DATA_OFFSETS['quaternion_z'])[0],
    )


def acquisition_metadata_from_scan_header(scan_header: TwixScanHeader) -> SiemensAcquisitionMetadata:
    """Convert a parsed Siemens scan header into acquisition metadata.

    Parameters
    ----------
    scan_header
        Parsed Siemens scan-header metadata.

    Returns
    -------
        Per-acquisition Siemens metadata.
    """
    read_dir, phase_dir, slice_dir = quaternion_to_directions(scan_header.quaternion)
    return SiemensAcquisitionMetadata(
        measurement_uid=scan_header.measurement_uid,
        scan_counter=scan_header.scan_counter,
        acquisition_time_stamp=scan_header.acquisition_time_stamp,
        physiology_time_stamp=scan_header.physiology_time_stamp,
        position=scan_header.position,
        read_dir=read_dir,
        phase_dir=phase_dir,
        slice_dir=slice_dir,
        patient_table_position=scan_header.patient_table_position,
        samples_in_scan=scan_header.samples_in_scan - scan_header.discard_pre - scan_header.discard_post,
        used_channels=scan_header.used_channels,
        discard_pre=scan_header.discard_pre,
        discard_post=scan_header.discard_post,
    )


def crop_channels(
    channels: list[np.ndarray],
    discard_pre: int,
    discard_post: int,
) -> list[np.ndarray]:
    """Crop Siemens channel data according to the scan-header cutoffs.

    Parameters
    ----------
    channels
        Channel-aligned complex sample arrays.
    discard_pre
        Number of scanner-side samples discarded before the kept readout.
    discard_post
        Number of scanner-side samples discarded after the kept readout.

    Returns
    -------
        Cropped channel arrays.
    """
    if not channels:
        return channels
    stop_index = channels[0].shape[0] - discard_post if discard_post else channels[0].shape[0]
    if discard_pre < 0 or discard_post < 0 or discard_pre > stop_index:
        raise ValueError('Invalid Siemens discard_pre/discard_post values in the scan header.')
    return [channel[discard_pre:stop_index] for channel in channels]


def quaternion_to_directions(
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Convert a Siemens quaternion into MRD direction vectors.

    Parameters
    ----------
    quaternion_wxyz
        Siemens quaternion in stored order.

    Returns
    -------
        Read, phase, and slice direction vectors.
    """
    x, y, z, w = quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rotation_matrix = (
        (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)),
        (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)),
        (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)),
    )
    phase_dir = normalize_direction((rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]))
    read_dir = normalize_direction((rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]))
    slice_dir = normalize_direction((rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]))
    return read_dir, phase_dir, slice_dir


def normalize_direction(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a direction vector.

    Parameters
    ----------
    direction
        Direction vector.

    Returns
    -------
        Normalized direction vector.
    """
    norm = math.sqrt(sum(value * value for value in direction))
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return (
        direction[0] / norm,
        direction[1] / norm,
        direction[2] / norm,
    )


def has_eval_info_bit(eval_info_mask: tuple[int, int], bit_index: int) -> bool:
    """Return whether a Siemens eval-info bit is set.

    Parameters
    ----------
    eval_info_mask
        Two-word Siemens eval-info bit mask.
    bit_index
        Zero-based bit index.

    Returns
    -------
        True when the requested bit is set.
    """
    word_index = bit_index // 32
    bit_offset = bit_index % 32
    if word_index >= len(eval_info_mask):
        return False
    return bool(eval_info_mask[word_index] & (1 << bit_offset))


def dma_length(raw_header: bytes) -> int:
    """Return the Siemens DMA record length from a raw scan header.

    Parameters
    ----------
    raw_header
        Raw scan-header bytes.

    Returns
    -------
        Siemens DMA record length.
    """
    return struct.unpack_from('<I', raw_header, 0)[0] & MDH_DMA_LENGTH_MASK


def read_exact(handle: BinaryIO, size: int) -> bytes:
    """Read exactly the requested number of bytes.

    Parameters
    ----------
    handle
        Open file handle.
    size
        Number of bytes to read.

    Returns
    -------
        Requested byte block.
    """
    data = handle.read(size)
    if len(data) < size:
        raise ValueError(f'Unexpected end of file while reading {size} bytes.')
    return data


def read_c_string(handle: BinaryIO) -> str:
    """Read a NUL-terminated Latin-1 string from a file handle.

    Parameters
    ----------
    handle
        Open file handle.

    Returns
    -------
        Decoded string.
    """
    characters = bytearray()
    while True:
        next_byte = handle.read(1)
        if next_byte == b'':
            return characters.decode('latin1', errors='ignore')
        if next_byte == b'\x00':
            return characters.decode('latin1', errors='ignore')
        characters.extend(next_byte)
        if len(characters) > MEASUREMENT_BUFFER_NAME_LIMIT * 1024:
            raise ValueError('Encountered an unexpectedly long Siemens C string.')


def read_c_string_from_bytes(data: bytes) -> str:
    """Decode a fixed-size Siemens C string field.

    Parameters
    ----------
    data
        Raw fixed-size field bytes.

    Returns
    -------
        Decoded string.
    """
    return data.split(b'\x00', 1)[0].decode('latin1', errors='ignore')


def align_to_32_bytes(handle: BinaryIO, *, base_offset: int) -> None:
    """Align a file handle to the next 32-byte boundary.

    Parameters
    ----------
    handle
        Open file handle.
    base_offset
        Absolute base offset for the current measurement.
    """
    relative_position = handle.tell() - base_offset
    if relative_position % 32:
        handle.seek(32 - (relative_position % 32), 1)


def first_present(*values: str | None) -> str | None:
    """Return the first non-empty string.

    Parameters
    ----------
    values
        Candidate string values.

    Returns
    -------
        First non-empty string, if available.
    """
    for value in values:
        if value:
            return value
    return None


def normalize_siemens_date(value: str | None) -> str | None:
    """Normalize a Siemens date string to ISO form when possible.

    Parameters
    ----------
    value
        Raw Siemens date string.

    Returns
    -------
        Normalized ISO date string.
    """
    if not value:
        return None
    digits = ''.join(character for character in value if character.isdigit())
    if len(digits) != 8:
        return value
    return f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}'


def normalize_siemens_time(value: str | None) -> str | None:
    """Normalize a Siemens time string to ISO-like form when possible.

    Parameters
    ----------
    value
        Raw Siemens time string.

    Returns
    -------
        Normalized time string.
    """
    if not value:
        return None
    cleaned_value = value.strip()
    if ':' in cleaned_value:
        return cleaned_value
    parts = cleaned_value.split('.', 1)
    digits = ''.join(character for character in parts[0] if character.isdigit())
    if len(digits) < 6:
        return cleaned_value
    normalized_value = f'{digits[:2]}:{digits[2:4]}:{digits[4:6]}'
    if len(parts) == 2 and parts[1]:
        return f'{normalized_value}.{parts[1]}'
    return normalized_value
