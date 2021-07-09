# -*- coding: utf-8 -*-

"""Top-level package for noise2inverse."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"


def __get_version():
    import os.path

    version_filename = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

from . import datasets
from . import fig
from . import noise
from . import tiffs
from . import tomo
from .unet import UNet
from .dncnn import DnCNN
