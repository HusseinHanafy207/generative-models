import generative_models
from generative_models.utils import get_device


def test_package_imports():
    assert generative_models.__version__ == "0.1.0"


def test_get_device():
    device = get_device()
    assert device.type in ("cpu", "cuda")
