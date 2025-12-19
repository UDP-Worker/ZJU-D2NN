import numpy as np

def read_from_camera() -> np.ndarray:
    return

def write_to_hdmi(picture: np.ndarray, channel: int) -> None:
    if channel == 1:
        assert picture.shape == # SLM1's pixel shape
    elif channel == 2:
        assert picture.shape == # SLM2's pixel shape
    else:
        raise NotImplementedError

    return

def sample_images(input_pictures: np.ndarray) -> np.ndarray:
    """
    Write images from dataset coordinately to hdmi1 (SLM1, the picture loader),
    and remain (SLM2, the optical neural network unchanged), meanwhile, collect
    images from camera when different pictures are loaded.

    return: np.ndarray with shape (batch_size, height, width, grayscale)
    """

    return