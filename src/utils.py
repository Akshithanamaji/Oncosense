import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cpu: bool = False):
    if not prefer_cpu and torch.cuda.is_available():
        try:
            # Verify CUDA context is alive and current to the thread
            _t = torch.zeros(1, device="cuda")
            del _t
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


class AverageMeter:
    """Tracks a running average of a metric (e.g. loss) over a training loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
