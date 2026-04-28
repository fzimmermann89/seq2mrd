"""Synthetic Siemens Twix test data helpers."""

from __future__ import annotations

import struct

SAMPLE_MEAS_HEADER = """
<XProtocol>{
  <ParamMap."Meas">{
    <ParamMap."DICOM">{
      <ParamString."tPatientName">{ "Ada Lovelace" }
      <ParamString."Manufacturer">{ "Siemens" }
      <ParamString."ManufacturersModelName">{ "Prisma" }
      <ParamString."StudyDate">{ "20260427" }
      <ParamString."StudyTime">{ "123456.789" }
      <ParamString."SeriesDate">{ "20260427" }
      <ParamString."SeriesTime">{ "123500.000" }
    }
  }
}
"""


def build_vb_file() -> bytes:
    """Build a synthetic VB Twix file."""
    measurement = build_measurement_preamble(SAMPLE_MEAS_HEADER)
    record = build_vb_record()
    return measurement + record


def build_vd_file(
    record_count: int = 1,
    *,
    discard_pre: int = 0,
    discard_post: int = 0,
    samples: tuple[complex, ...] = (1 + 2j, 3 + 4j),
) -> bytes:
    """Build a synthetic single-measurement VD Twix file."""
    measurement = build_measurement_preamble(SAMPLE_MEAS_HEADER)
    measurement_offset = 8 + 152
    records = b''.join(
        build_vd_record(
            scan_counter=4 + index,
            is_last_scan=index == record_count - 1,
            discard_pre=discard_pre,
            discard_post=discard_post,
            samples=samples,
        )
        for index in range(record_count)
    )
    measurement_length = len(measurement) + len(records)
    raid_header = struct.pack('<II', 0, 1)
    raid_entry = bytearray(152)
    struct.pack_into('<IIQQ', raid_entry, 0, 1, 1, measurement_offset, measurement_length)
    write_c_string_into(raid_entry, 24, 64, 'Ada Lovelace')
    write_c_string_into(raid_entry, 88, 64, 'PulseqProtocol')
    return raid_header + bytes(raid_entry) + measurement + records


def build_vd_multi_measurement_file(first_header: str, second_header: str) -> bytes:
    """Build a VD Twix file with two measurements."""
    first_measurement = build_measurement_preamble(first_header) + build_vd_record(scan_counter=1, is_last_scan=True)
    second_measurement = build_measurement_preamble(second_header) + build_vd_record(scan_counter=4, is_last_scan=True)
    first_offset = 8 + 2 * 152
    second_offset = first_offset + len(first_measurement)

    raid_header = struct.pack('<II', 0, 2)
    first_entry = bytearray(152)
    struct.pack_into('<IIQQ', first_entry, 0, 1, 1, first_offset, len(first_measurement))
    write_c_string_into(first_entry, 24, 64, 'First Measurement')
    write_c_string_into(first_entry, 88, 64, 'AdjCoilSens')

    second_entry = bytearray(152)
    struct.pack_into('<IIQQ', second_entry, 0, 2, 2, second_offset, len(second_measurement))
    write_c_string_into(second_entry, 24, 64, 'Ada Lovelace')
    write_c_string_into(second_entry, 88, 64, 'PulseqProtocol')

    return raid_header + bytes(first_entry) + bytes(second_entry) + first_measurement + second_measurement


def build_measurement_preamble(header_text: str) -> bytes:
    """Build a Siemens measurement preamble with one `Meas` buffer."""
    header_bytes = header_text.encode('latin1')
    preamble = bytearray()
    preamble.extend(struct.pack('<II', 128, 1))
    preamble.extend(b'Meas\x00')
    preamble.extend(struct.pack('<I', len(header_bytes)))
    preamble.extend(header_bytes)
    while len(preamble) % 32:
        preamble.append(0)
    return bytes(preamble)


def build_vb_record() -> bytes:
    """Build one synthetic VB record."""
    samples_in_scan = 2
    record = bytearray(128 + samples_in_scan * 8)
    struct.pack_into('<I', record, 0, 128 + samples_in_scan * 8)
    struct.pack_into('<i', record, 4, 17)
    struct.pack_into('<I', record, 8, 3)
    struct.pack_into('<I', record, 12, 654321)
    struct.pack_into('<I', record, 16, 222)
    struct.pack_into('<II', record, 20, 1 << 0, 0)
    struct.pack_into('<H', record, 28, samples_in_scan)
    struct.pack_into('<H', record, 30, 1)
    struct.pack_into('<fff', record, 96, 4.0, 5.0, 6.0)
    struct.pack_into('<ffff', record, 108, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into('<H', record, 126, 7)
    struct.pack_into('<ffff', record, 128, 1.0, 2.0, 3.0, 4.0)
    return bytes(record)


def build_vd_record(
    scan_counter: int,
    *,
    is_last_scan: bool,
    discard_pre: int = 0,
    discard_post: int = 0,
    samples: tuple[complex, ...] = (1 + 2j, 3 + 4j),
) -> bytes:
    """Build one synthetic VD record."""
    samples_in_scan = len(samples)
    record = bytearray(192 + 32 + samples_in_scan * 8)
    struct.pack_into('<I', record, 0, 192 + 32 + samples_in_scan * 8)
    struct.pack_into('<i', record, 4, 19)
    struct.pack_into('<I', record, 8, scan_counter)
    struct.pack_into('<I', record, 12, 123456)
    struct.pack_into('<I', record, 16, 333)
    struct.pack_into('<i', record, 24, 10)
    struct.pack_into('<i', record, 28, 20)
    struct.pack_into('<i', record, 32, 30)
    struct.pack_into('<II', record, 40, (1 << 0) if is_last_scan else 0, 0)
    struct.pack_into('<H', record, 48, samples_in_scan)
    struct.pack_into('<H', record, 50, 1)
    struct.pack_into('<H', record, 80, discard_pre)
    struct.pack_into('<H', record, 82, discard_post)
    struct.pack_into('<fff', record, 100, 1.5, 2.5, 3.5)
    struct.pack_into('<ffff', record, 112, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into('<I', record, 192, 32 + samples_in_scan * 8)
    struct.pack_into('<H', record, 216, 0)
    for sample_index, sample in enumerate(samples):
        struct.pack_into('<ff', record, 224 + sample_index * 8, float(sample.real), float(sample.imag))
    return bytes(record)


def write_c_string_into(buffer: bytearray, offset: int, size: int, value: str) -> None:
    """Write a fixed-size Siemens C string field."""
    encoded_value = value.encode('latin1')
    buffer[offset : offset + size] = encoded_value[:size].ljust(size, b'\x00')
