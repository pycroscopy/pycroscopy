"""
The Pycroscopy package.

As of March 5th, the previous iteration of pycroscopy has been archived
and is now on the 'legacy' branch. In its place has arisen a new, improved
vision for px as discussed in issue 245 (https://github.com/pycroscopy/pycroscopy/issues/245)

The revised pycroscopy (px) package is currently on branch 'phoenix'
This revised package is more generic than the traditional px.
It should also in time be much more powerful, and will accept sidpy dataset objects
by default, in line with the ethos of the px ecosystem.

Submodules
----------
.. autosummary::
    :toctree: _autosummary
"""

from . import corr
from . import fft
from . import image
from . import learn
from . import viz
from . import signal
from . import stats

from .__version__ import version as __version__
from .__version__ import time as __time__
