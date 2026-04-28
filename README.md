# seq2mrd

`seq2mrd` builds ISMRMRD acquisitions from a pulseq `.seq` file and enriches them with vendor raw data.

The package is designed around an in-memory workflow:

- pulseq defines the acquisition structure
- vendor raw data provides samples and scanner metadata
- the result can be used directly in Python or written to an `.mrd` file

## Status

The current implementation is Siemens-first.

- Siemens Twix input is implemented
- NumPy `.npy` input is implemented for simple in-memory style raw data
- Bruker and GE entry points exist, but currently raise `NotImplementedError`

## Design

The conversion is split into two steps.

### 1. Build an MRD skeleton from pulseq

`MRDSkeleton.from_seq(...)` reads the `.seq` file and creates:

- an ISMRMRD header
- one acquisition per ADC event
- embedded trajectory samples in each acquisition
- indices from pulseq labels when available

This step uses pulseq as the source of truth for:

- acquisition count and order
- trajectory
- encoding and reconstruction geometry
- sequence parameters such as `TR`, `TE`, and `TI`

### 2. Fill the skeleton from vendor raw data

Vendor-specific `RawSource` implementations enrich the skeleton with:

- complex raw k-space samples
- acquisition timestamps
- orientation and position
- table position
- study and system metadata when available

For Siemens, this is done with:

```python
from seq2mrd import MRDSkeleton, SiemensRaw

skeleton = MRDSkeleton.from_seq('sequence.seq')
skeleton = SiemensRaw('meas.dat')(skeleton)
```

For simple raw data stored as a NumPy array, use:

```python
from seq2mrd import MRDSkeleton, NumpyRaw

skeleton = MRDSkeleton.from_seq('sequence.seq')
skeleton = NumpyRaw('raw.npy')(skeleton)
```

The `.npy` file is expected to contain a complex array with shape `(n_adcs, n_coils, n_readout)`.

## Supported seq definitions

The pulseq path currently reads these definitions when present:

- `FOV`
- `ReconFOV`
- `EncodingMatrix`
- `ReconMatrix`
- `TR`
- `TE`
- `TI`
- `ReadoutOversamplingFactor`
- `TrajectoryType`

Fallbacks are used when optional definitions are missing:

- `ReconFOV` falls back to `FOV`
- `EncodingMatrix` falls back to ADC and label heuristics
- `ReconMatrix` falls back to encoding-matrix heuristics
- if no pulseq labels are present, `LIN` is generated within ADC-frequency classes and the class ID is stored in `user_int[7]`

## Library usage

### Convenience functions

The simplest interface returns parsed ISMRMRD objects directly:

```python
from seq2mrd import convert_siemens

header, acquisitions = convert_siemens('sequence.seq', 'meas.dat')
```

For NumPy raw input:

```python
from seq2mrd import convert_numpy

header, acquisitions = convert_numpy('sequence.seq', 'raw.npy')
```

This can be passed directly into [MRpro](https://github.com/PTB-MR/mrpro) or [mr2](https://github.com/fzimmermann89/mr2):

```python
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from seq2mrd import convert_siemens

header, acquisitions = convert_siemens('sequence.seq', 'meas.dat')
kdata = KData.from_ismrmrd(header, acquisitions, trajectory=KTrajectoryIsmrmrd())
```

The `mr2` integration path is still work in progress and is not yet merged upstream.

### Explicit skeleton workflow

If you want to work with the intermediate in-memory representation:

```python
from seq2mrd import MRDSkeleton, SiemensRaw

skeleton = MRDSkeleton.from_seq('sequence.seq')
skeleton = SiemensRaw('meas.dat')(skeleton)

header = skeleton.header
acquisitions = skeleton.acquisitions
```

### Writing an `.mrd` file

`MRDSkeleton` can also be written to disk:

```python
from seq2mrd import MRDSkeleton, SiemensRaw

skeleton = SiemensRaw('meas.dat')(MRDSkeleton.from_seq('sequence.seq'))
skeleton.write('output.mrd')
```

## Command-line usage

After installation, the package provides a `seq2mrd` command.

### Siemens

```bash
seq2mrd siemens --seq sequence.seq --raw meas.dat --output output.mrd
```

### NumPy

```bash
seq2mrd numpy --seq sequence.seq --raw raw.npy --output output.mrd
```

### Custom dataset name

```bash
seq2mrd siemens --seq sequence.seq --raw meas.dat --output output.mrd --dataset-name dataset_2
```

The CLI writes an ISMRMRD file and uses the same conversion path as the Python API.

## Installation

Install the package with `pip`:

```bash
pip install git+https://github.com/fzimmermann89/seq2mrd.git
```

The command-line entry point is installed as part of the package.
