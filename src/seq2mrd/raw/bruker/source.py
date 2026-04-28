"""Bruker raw data source."""

from seq2mrd.raw.base import RawSource
from seq2mrd.skeleton import MRDSkeleton


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
        raise NotImplementedError('Bruker raw data support has not been implemented yet.')
