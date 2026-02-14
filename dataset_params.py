"""Dataset-specific camera, sensor, and scene parameters.

Groups all per-dataset constants that were previously stored as mutable
module-level globals in ``globals.py``.  Each dataset has a factory
method that returns a pre-filled instance.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetParams:
    """Camera, sensor, and scene parameters for a single dataset."""

    f: float = 0.0              # focal length (m or unitless for MobileDepth)
    D: float = 0.0              # aperture diameter (m or unitless for MobileDepth)
    Zf: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    ps: float = 0.0             # pixel size (m or unitless for MobileDepth)
    min_Z: float = 0.0          # scene depth range lower bound
    max_Z: float = 0.0          # scene depth range upper bound
    thresh: float = 0.0         # forward model CoC threshold

    @staticmethod
    def for_NYUv2():
        """Pre-filled parameters for the NYUv2 indoor RGB-D dataset."""
        return DatasetParams(
            f=50e-3,
            D=50e-3 / 8,
            Zf=np.array([1, 1.5, 2.5, 4, 6], dtype=np.float32),
            ps=1.2e-5,
            min_Z=0.1,
            max_Z=10.0,
            thresh=2,
        )

    @staticmethod
    def for_MobileDepth():
        """Partially filled parameters for MobileDepth.

        ``f``, ``D``, and ``Zf`` are set by the dataset loader from
        per-scene calibration files.
        """
        return DatasetParams(
            thresh=0.1,
            ps=0.75 * 2,
            min_Z=1,
            max_Z=800,
        )

    @staticmethod
    def for_Make3D():
        """Partially filled parameters for Make3D.

        ``f``, ``D``, and ``ps`` are set by the dataset loader from
        EXIF metadata and image dimensions.
        """
        return DatasetParams(
            Zf=np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.float32),
            min_Z=0.01,
            max_Z=80,
            thresh=0.5,
        )
