import sys, os
sys.path.append(os.path.dirname(__file__))
# from .asset.erm import ERMTrainer
# from .asset.ttt import TestTimeTrainer

from .scripts.ttt_main import TTTModel
from .scripts.erm_main import ERMModel
from .scripts.coral_main import CORALModel
from .scripts.dann_main import SERVER_DANNModel
from .scripts.irm_main import IRMModel
from .scripts.mixup_main import MIXUPModel
from .scripts.mask_main import MASKModel
from .scripts.groupdro_main import GROUPDROModel

