"""Convenience conversion helpers."""

import ismrmrd

from seq2mrd.raw.bruker import BrukerRaw
from seq2mrd.raw.ge import GeRaw
from seq2mrd.raw.numpy import NumpyRaw
from seq2mrd.raw.siemens import SiemensRaw
from seq2mrd.skeleton import MRDSkeleton

MrdHeader = ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader


def convert_siemens(
    seq_path: str,
    raw_path: str,
) -> tuple[MrdHeader, list[ismrmrd.Acquisition]]:
    """Convert Siemens raw data and pulseq into ISMRMRD header and acquisitions."""
    skeleton = SiemensRaw(raw_path)(MRDSkeleton.from_seq(seq_path))
    return skeleton.header, skeleton.acquisitions


def convert_bruker(
    seq_path: str,
    raw_path: str,
) -> tuple[MrdHeader, list[ismrmrd.Acquisition]]:
    """Convert Bruker raw data and pulseq into ISMRMRD header and acquisitions."""
    skeleton = BrukerRaw(raw_path)(MRDSkeleton.from_seq(seq_path))
    return skeleton.header, skeleton.acquisitions


def convert_ge(
    seq_path: str,
    raw_path: str,
) -> tuple[MrdHeader, list[ismrmrd.Acquisition]]:
    """Convert GE raw data and pulseq into ISMRMRD header and acquisitions."""
    skeleton = GeRaw(raw_path)(MRDSkeleton.from_seq(seq_path))
    return skeleton.header, skeleton.acquisitions


def convert_numpy(
    seq_path: str,
    raw_path: str,
) -> tuple[MrdHeader, list[ismrmrd.Acquisition]]:
    """Convert NumPy raw data and pulseq into ISMRMRD header and acquisitions."""
    skeleton = NumpyRaw(raw_path)(MRDSkeleton.from_seq(seq_path))
    return skeleton.header, skeleton.acquisitions
