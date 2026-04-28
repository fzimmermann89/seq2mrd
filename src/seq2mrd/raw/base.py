"""Base classes for raw data sources."""

from abc import ABC, abstractmethod
from pathlib import Path

from seq2mrd.skeleton import MRDSkeleton


class RawSource(ABC):
    """Base class for vendor-specific raw data enrichers."""

    def __init__(self, raw_path: str | Path) -> None:
        """Store the raw data location.

        Parameters
        ----------
        raw_path
            Path to the vendor-specific raw data source.
        """
        self.raw_path = Path(raw_path)

    @abstractmethod
    def __call__(self, skeleton: MRDSkeleton) -> MRDSkeleton:
        """Fill a pulseq-derived skeleton with raw data and vendor metadata.

        Parameters
        ----------
        skeleton
            Skeleton created from the pulseq file.

        Returns
        -------
            Updated skeleton.
        """
