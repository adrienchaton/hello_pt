# examples around optim and autograd


import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size, n_features, n_classes = 8, 4, 3
cuda_device = 0  # compute on 1st visible GPU if possible
device = torch.device(f"cuda:{cuda_device}" if (torch.cuda.is_available() and cuda_device>=0) else "cpu")

dummy_x = torch.randn((batch_size, n_features), device=device, dtype=torch.float32)  # instantiate random data on device
dummy_y = torch.randint(low=0, high=n_classes, size=(batch_size, ), device=device, dtype=torch.long)

########################################################################################################################
## setting up a simple model, optimizer, loss and perform one update

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.act = nn.Softmax(dim=1)  # to compute probabilities over the classes dimension
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', weight=None, ignore_index=-100)
        # TODO: weight can used to e.g. compensate class imbalance
        #   and ignore_index can be used to fill-up missing values to be ignored, or as padding value
    def forward(self, x):
        return self.layer(x)
    def compute_loss(self, x, y):
        y_hat = self.forward(x)
        y_loss = self.loss_fn(y_hat, y) # here, CrossEntropyLoss already handles scaling the logits before computing NLL
        return {"y_hat": self.act(y_hat), "y_loss": y_loss}
    @torch.inference_mode()  # this automatically disable gradients and reduce calculations to speed-up inference
    def predict(self, x):
        y_hat = self.forward(x)
        return {"y_hat": self.act(y_hat)}

model1 = LinearModel(n_features, n_classes).to(device)
optim1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=0.01)
print(f"\nmodel1: with {sum(p.numel() for p in model1.parameters() if p.requires_grad)} trainable parameters and "
      f"{sum(p.numel() for p in model1.parameters() if not p.requires_grad)} fixed parameters")
# model1: with 15 trainable parameters and 0 fixed parameters

def try_optim_step(model, optim, x, y):
    try:
        loss = model.compute_loss(x, y)["y_loss"]
        optim.zero_grad()  # ensure all gradients are cleared before doing an update step
        loss.backward()
        print("forward loss", loss)  # you can see the trace of grad_fn=<NllLossBackward0>
        print(f"gradients with norm={torch.sum(torch.abs(model.layer.weight.grad)).item()}")
        optim.step()  # this will act on the provided model.parameters() which have gradients
        print(f"optim state = {optim.state_dict()}")  # here you see e.g. moving average values due to SGD with momentum
        # TODO: usually, both the model and the optimizer states are saved, to allow e.g. resuming training
        optim.zero_grad()
        print(f"clearing gradients after optimization step, gradients={model.layer.weight.grad}")  # gradients=None
    except RuntimeError:
        # we get "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        optim.zero_grad()
        print(f"could not perform optimization step")

model1.train()
print("\nmodel1: step in train mode")
try_optim_step(model1, optim1, dummy_x, dummy_y)

model1.eval()  # this doesn't mean gradients are disabled!
print("\nmodel1: step in eval mode")
try_optim_step(model1, optim1, dummy_x, dummy_y)

print("\nmodel1: step in inference mode")
with torch.inference_mode():  # this effectively disable everything related to autograd
    # preferred over torch.no_grad()
    try_optim_step(model1, optim1, dummy_x, dummy_y)  # --> could not perform optimization step

########################################################################################################################
## more "custom" model, checking where gradients are properly computed

class FrankensteinModel(LinearModel):
    def __init__(self, input_size, output_size):
        super(FrankensteinModel, self).__init__(input_size, output_size)
        self.register_buffer("a_buffer", torch.randn(input_size, dtype=torch.float32), persistent=True)
        self.learned_p = nn.Parameter(torch.randn(input_size, dtype=torch.float32), requires_grad=True)
        self.fixed_p = nn.Parameter(torch.randn(input_size, dtype=torch.float32), requires_grad=False)
    def forward(self, x):
        x = x+self.a_buffer
        x = x+self.learned_p
        x = x+self.fixed_p
        return self.layer(x)
    def gradient_check(self, optim, x, y):
        loss = self.compute_loss(x, y)["y_loss"]
        optim.zero_grad()
        loss.backward()
        for pname, param in self.named_parameters():
            if param.grad is None:
                print(f"{pname} has no gradients")
            else:
                print(f"{pname} has gradients with norm {torch.sum(torch.abs(param.grad)).item()}")
        optim.zero_grad()
# TODO: "gradient_check" can be a good sanity check to make sure all init layers are training
#   and check their respective gradient magnitudes

model2 = FrankensteinModel(n_features, n_classes).to(device)
optim2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
print(f"\nmodel2: with {sum(p.numel() for p in model2.parameters() if p.requires_grad)} trainable parameters and "
      f"{sum(p.numel() for p in model2.parameters() if not p.requires_grad)} fixed parameters")
# model2: with 19 trainable parameters and 4 fixed parameters

print("\ngradient check")
model2.gradient_check(optim2, dummy_x, dummy_y)  # --> fixed_p has no gradients

print("\ngradient check after freezing learned_p")
model2.learned_p.requires_grad = False
# it can be useful for e.g. fine-tuning to dynamically freeze/un-freeze some parameters
model2.gradient_check(optim2, dummy_x, dummy_y)  # --> learned_p has no gradients (too)

print("\ngradient check after adding fixed_p2 to model")
model2.fixed_p2 = nn.Parameter(torch.randn(1, device=device, dtype=torch.float32), requires_grad=False)
# alternatively, model2.add_module("another_layer", nn.Linear(n_features, n_features))
model2.to(device)  # make sure all params. are on the same device
model2.gradient_check(optim2, dummy_x, dummy_y)  # --> fixed_p2 has no gradients (too)
# TODO: if we wished to add new trainable parameters, we would need e.g.
#   - optim.add_param_group({'params': the_new_parameters})
#   - modify the forward method to apply the parameters in the computation
# TODO: in some cases we don't want an optimizer to update all parameters, e.g. GANs
#   and we also need to detach some variables from the autograd graph, e.g. generator outputs in the discriminator loss
y_loss = model2.compute_loss(dummy_x, dummy_y)["y_loss"]
print(y_loss)  # tensor(0.7223, device='cuda:0', grad_fn=<NllLossBackward0>)
print(y_loss.detach())  # tensor(0.7223, device='cuda:0')
print(y_loss.detach().cpu())  # tensor(0.7223)
print(y_loss.detach().cpu().numpy())  # 0.7223418



