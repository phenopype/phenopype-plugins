#%% imports

from importlib import util, metadata

__version__ = metadata.version("phenopype_plugins")

assert util.find_spec("phenopype"), "phenopype-plugins will not work without the main package"

from .plugins import segmentation, measurement
