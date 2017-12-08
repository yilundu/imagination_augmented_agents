from __future__ import print_function, absolute_import

import logging
import os

thisfile = os.path.abspath(__file__)

try:
    from . import image
except ImportError:
    logging.error('{}: failed to import "image"'.format(thisfile))

try:
    from . import shell
except ImportError:
    logging.error('{}: failed to import "shell"'.format(thisfile))

#try:
#    from . import torch
#except ImportError:
#    logging.error('{}: failed to import "torch"'.format(thisfile))

try:
    from . import video
except ImportError:
    logging.error('{}: failed to import "video"'.format(thisfile))

try:
    from .logger import *
except ImportError:
    logging.error('{}: failed to import "logger"'.format(thisfile))

try:
    from .util_math import *
except ImportError:
    logging.error('{}: failed to import "math"'.format(thisfile))
