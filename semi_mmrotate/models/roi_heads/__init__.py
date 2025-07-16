from .rfrcnn_roi_head import RotatedFRCNNRoIHead
from .orcnn_roi_head import ORCNNRoIHead
# from .gi_roi_head import GIRoIHead
# from .gi_roi_head_0710 import GIRoIHead
# from .gi_roi_head_0711 import GIRoIHead
from .gi_roi_head_0712 import GIRoIHead

__all__ = [
    'RotatedFRCNNRoIHead', 'ORCNNRoIHead', 'GIRoIHead'
]
