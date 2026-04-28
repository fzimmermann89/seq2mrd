"""GE raw data source."""

from seq2mrd.raw.base import RawSource
from seq2mrd.skeleton import MRDSkeleton


class GeRaw(RawSource):
    """GE raw data enricher."""

    def __call__(self, skeleton: MRDSkeleton) -> MRDSkeleton:
        """Fill a pulseq-derived skeleton with GE raw data.

        Parameters
        ----------
        skeleton
            Skeleton created from the pulseq file.

        Returns
        -------
            Updated skeleton.
        """
        raise NotImplementedError('GE raw data support has not been implemented yet.')
