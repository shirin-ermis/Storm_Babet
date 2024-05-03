"""babet is a package to aid the extreme event attribution of Storm Babet (Oct 2023).

It contains functionality for importing and filtering data.

"""
# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import main classes
from .data import Data    # noqa
from .met import Met      # noqa