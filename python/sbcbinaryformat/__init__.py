"""
Contains:
* Streamer: which reads a file and either creates a streamer
  or saved everything to RAM. No multiprocessing features.
* TarStreamer: which reads a file from a tarball
* Writer: which creates SBC binary files given the passed parameters.
Very simplistic and does not have any multiprocessing features.
"""

from .streamer import Streamer
from .tarstreamer import TarStreamer
from .writer import Writer
from importlib.metadata import version, PackageNotFoundError

# expose package version
try:
    __version__ = version("sbcbinaryformat")
except PackageNotFoundError:
    __version__ = "0.0.0"
