"""NumPy raw data source."""

from __future__ import annotations

import numpy as np

from seq2mrd.raw.base import RawSource
from seq2mrd.skeleton import MRDSkeleton


class NumpyRaw(RawSource):
    """NumPy raw data enricher."""

    def __call__(self, skeleton: MRDSkeleton) -> MRDSkeleton:
        """Fill a pulseq-derived skeleton with `.npy` raw data.

        Parameters
        ----------
        skeleton
            Skeleton created from the pulseq file.

        Returns
        -------
            Updated skeleton.
        """
        raw_data = np.asarray(np.load(self.raw_path), dtype=np.complex64)
        if raw_data.ndim != 3:
            raise ValueError('NumPy raw data must have shape (n_adcs, n_coils, n_readout).')
        if raw_data.shape[0] != len(skeleton.acquisitions):
            raise ValueError(
                'The NumPy raw data contains a different number of ADC readouts than the pulseq skeleton. '
                f'Got {raw_data.shape[0]} readouts and {len(skeleton.acquisitions)} acquisitions.',
            )

        for acquisition, readout in zip(skeleton.acquisitions, raw_data, strict=True):
            if readout.shape[1] != acquisition.number_of_samples:
                raise ValueError(
                    'The NumPy raw data readout length does not match the pulseq ADC sample count. '
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

        return skeleton
