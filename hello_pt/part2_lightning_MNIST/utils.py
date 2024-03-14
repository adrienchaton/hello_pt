

import os
from pathlib import Path
import yaml
import torch
from lightning.pytorch.callbacks import Callback


class MNIST_CST:
    def __init__(self):
        self.path_to_mnist = os.path.join(Path(__file__).parent, "artefacts")
        self.MNISThw = 28
        self.n_classes = 10

def load_config(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        config = yaml.safe_load(f)
    return config

def try_checking_process_rank0():
    try:
        return torch.distributed.get_rank() == 0
    except:  # fallback in case we are not using distributed
        return True

class PrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("\nTraining is starting (╯°□°)╯︵ ┻━┻") if try_checking_process_rank0() else None
    def on_train_end(self, trainer, pl_module):
        print("\nTraining is ending ♩¨̮(ง ˙˘˙ )ว♩¨̮") if try_checking_process_rank0() else None

class GradientCheckCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if try_checking_process_rank0():
            x, y = next(iter(pl_module.train_data))
            logits = pl_module.model.forward(x.to(pl_module.device, non_blocking=True))
            loss = pl_module.loss_fn(logits, y.to(pl_module.device, non_blocking=True))
            loss.backward()
            tot_grad = 0
            for pname, param in pl_module.model.named_parameters():
                if param.grad is None:
                    print(f"{pname} has no gradients")
                else:
                    pgrad = torch.sum(torch.abs(param.grad)).item()
                    tot_grad += pgrad
                    print(f"{pname} has gradients with norm {pgrad}")
            print(f"total sum of gradients = {tot_grad}")
    def on_train_end(self, trainer, pl_module):
        self.on_train_start(trainer, pl_module)

# FIXME: getting some unexpected errors with the callback below
# class TestingCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         results = trainer.test(model=pl_module, dataloaders=pl_module.test_data, verbose=True)
#         print(f"\ntesting metrics\n{results}")
#     def on_train_end(self, trainer, pl_module):
#         self.on_train_start(trainer, pl_module)

