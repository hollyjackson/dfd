"""
Configuration classes for different datasets and experiments.

Each config class contains all hyperparameters and settings for running
the depth-from-defocus pipeline on a specific dataset that were used in
the main paper experiments.
"""

from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Base configuration with common default values."""

    # AIF initialization parameters
    aif_lambda: float = 0.05
    aif_sharpness_measure: str = 'sobel_grad'

    # Coordinate descent parameters
    nesterov_first: bool = False

    # Experiment settings
    experiment_folder: str = 'experiments'
    show_plots: bool = False
    save_plots: bool = False
    verbose: bool = False

    # Dataset directory
    data_dir: str = 'data'

    def get_experiment_name(self, example_name: str) -> str:
        """Generate experiment name from configuration."""
        raise NotImplementedError("Subclasses must implement get_experiment_name")


@dataclass
class NYUv2Config(BaseConfig):
    """Configuration for NYUv2 dataset."""

    # Windowed MSE parameters
    use_windowed_mse: bool = False

    # Coordinate descent parameters
    num_epochs: int = 40
    num_z: int = 100
    t_0: int = 200
    alpha: float = 1.05

    experiment_folder: str = 'experiments/nyuv2'

    def get_experiment_name(self, split: str, image_number: str) -> str:
        """Generate experiment name for NYUv2."""
        name = 'nyuv2-'
        name += split + '-'
        if self.use_windowed_mse:
            name += f"windowed{self.window_size}-"
        name += f"img{image_number}"
        return name


@dataclass
class Make3DConfig(BaseConfig):
    """Configuration for Make3D dataset."""

    # Windowed MSE parameters
    use_windowed_mse: bool = True
    window_size: int = 5

    # Coordinate descent parameters
    num_epochs: int = 5
    num_z: int = 100
    t_0: int = 10
    alpha: float = 2.0

    experiment_folder: str = 'experiments/make3d'

    def get_experiment_name(self, split: str, img_name: str) -> str:
        """Generate experiment name for Make3D."""
        name = 'make3d-'
        name += split + '-'
        if self.use_windowed_mse:
            name += f"windowed{self.window_size}-"
        name += img_name.split("img-")[1].split(".jpg")[0]
        return name


@dataclass
class MobileDepthConfig(BaseConfig):
    """Configuration for Mobile Depth dataset."""

    # Windowed MSE parameters
    use_windowed_mse: bool = True
    window_size: int = 50

    # Coordinate descent parameters
    num_epochs: int = 5
    num_z: int = 200
    t_0: int = 10
    alpha: float = 2.0

    experiment_folder: str = 'experiments/mobiledepth'

    # Valid example names
    valid_examples: tuple = (
        "keyboard", "bottles", "fruits", "metals", "plants",
        "telephone", "window", "largemotion", "smallmotion",
        "zeromotion", "balls"
    )

    def get_experiment_name(self, example_name: str) -> str:
        """Generate experiment name for Mobile Depth."""
        name = 'mobile-depth-'
        if self.use_windowed_mse:
            name += f"windowed{self.window_size}-"
        name += example_name
        return name


# Pre-instantiated configs for easy import
NYUV2 = NYUv2Config()
MAKE3D = Make3DConfig()
MOBILE_DEPTH = MobileDepthConfig()
