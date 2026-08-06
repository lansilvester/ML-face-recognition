from .dashboard import DashboardView
from .register import RegisterView
from .train import TrainView
from .recognize import RecognizeView

VIEWS = {
    "dashboard": DashboardView,
    "register": RegisterView,
    "train": TrainView,
    "recognize": RecognizeView,
}
