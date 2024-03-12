# examples to show how to use torch.nn and some of the pitfalls


import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size, n_features, n_classes, x_max = 8, 4, 3, 7.9  # as in the previous iris_dataset
n_hidden_layers, hidden_size = 2, 16  # your first model configuration! ;)
cuda_device = 0  # compute on 1st visible GPU if possible
device = torch.device(f"cuda:{cuda_device}" if (torch.cuda.is_available() and cuda_device>=0) else "cpu")
# TODO: you can set (relative) visible devices through the environment variable CUDA_VISIBLE_DEVICES
#   but using os.environ["CUDA_VISIBLE_DEVICES"] doesn't work after the python codes have been called

########################################################################################################################
## 0.A) NN block with a linear layer

class LinearBlock_A(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearBlock_A, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.BatchNorm1d(output_size)
        self.act = nn.LeakyReLU()
        self.apply(self._init_p)  # TODO: customize init, or comment to use defaults
    def _init_p(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)  # TODO: note that usually, operations ending with "_" are in-place
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x
# TODO: further customize with e.g. dropout "regularization"

model1 = LinearBlock_A(n_features, hidden_size).to(device)  # instantiate model on device
input1 = torch.randn((batch_size, n_features), device=device, dtype=torch.float32)  # instantiate random data on device
output1 = model1(input1)
print(f"\nmodel1: forwarded input1 of shape {input1.shape} to output1 of shape {output1.shape}")
# model1: forwarded input1 of shape torch.Size([8, 4]) to output1 of shape torch.Size([8, 16])
# TODO: here the output is some intermediate/hidden features

def print_trainable_parameters(model, verbose=False):
    trainable_params = 0
    all_param = 0
    for pname, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if verbose:
            print(f"{'trainable' if param.requires_grad else 'freeze'}\t{pname} of size {param.numel()}")
    print(f"trainable params: {trainable_params} || "
          f"all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}")

print_trainable_parameters(model1, verbose=True)
# trainable       linear.weight of size 64
# trainable       linear.bias of size 16
# trainable       norm.weight of size 16
# trainable       norm.bias of size 16
# trainable params: 112 || all params: 112 || trainable%: 100.00
# TODO: notice that all parameters are trainable and the activation doesn't have any learned parameters

print(model1.state_dict().keys())  # TODO: notice the added "buffers" for tracking norm statistics
# ['linear.weight', 'linear.bias', 'norm.weight', 'norm.bias', 'norm.running_mean', 'norm.running_var', 'norm.num_batches_tracked']
torch.save(model1.state_dict(), os.path.join(Path(__file__).parent, "artefacts", "model1.pt"))
model1.apply(model1._init_p)
print(f"same output after random init., {torch.equal(output1, model1(input1))}")  # False
model1.load_state_dict(torch.load(os.path.join(Path(__file__).parent, "artefacts", "model1.pt"), map_location=device))
print(f"same output after restoring ckpt, {torch.equal(output1, model1(input1))}")  # True

########################################################################################################################
## 0.B) linear block with a fixed scaling factor

class LinearBlock_B(LinearBlock_A):
    def __init__(self, input_size, output_size, scale_value=1.):
        super(LinearBlock_B, self).__init__(input_size, output_size)
        self.register_buffer("scale_value", torch.tensor(scale_value, dtype=torch.float32), persistent=True)
        # buffers are persistent by default but not considered as model parameters
    def forward(self, x):
        x = x/self.scale_value
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x

model2 = LinearBlock_B(n_features, hidden_size, scale_value=x_max).to(device)
output2 = model2(input1)
assert all(d1==d2 for d1, d2 in zip(output1.shape, output2.shape))
print("\nmodel2: state dict after registering buffer for scale_value")
print(model2.state_dict().keys())  # TODO: notice the added buffer for scale_value
print_trainable_parameters(model2, verbose=False)  # unchanged, i.e. trainable params: 112 || all params: 112 || trainable%: 100.00

########################################################################################################################
## 0.C) sequential block with learned scaling factors for each input dimension (for the example ... "useless" in practice)

class LinearBlock_C(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearBlock_C, self).__init__()
        self.block = nn.Sequential(nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.LeakyReLU())
        # since we don't need to modify intermediate values, let's pack all layers into a sequential container
        self.learned_scaling = nn.Parameter(torch.ones(input_size, dtype=torch.float32), requires_grad=True)
        # the parameter is persistent and by default the gradients will be tracked for training
        # there are other ways to add parameters, e.g. register_parameter, register_module, add_module
    def forward(self, x):
        scale_value = F.softplus(self.learned_scaling)  # TODO: take care of e.g. zero division when training params
        x = x/scale_value  # TODO: note that scale_value is automatically broadcast to the batch size of x
        x = self.block(x)
        return x

model3 = LinearBlock_C(n_features, hidden_size).to(device)
output3 = model3(input1)
assert all(d1==d3 for d1, d3 in zip(output1.shape, output3.shape))
print("\nmodel3: state dict after adding parameter for learned_scaling")
print(model3.state_dict().keys())  # TODO: notice the added parameter for learned_scaling
print_trainable_parameters(model3, verbose=False)  # trainable params: 116 || all params: 116 || trainable%: 100.00

########################################################################################################################
## 1) MLP classifier

class MLP_classifier(nn.Module):
    def __init__(self, input_size, output_size, n_hiddens, d_hiddens):
        super(MLP_classifier, self).__init__()
        layers = [LinearBlock_A(input_size, d_hiddens)]  # the input layer
        layers += [LinearBlock_A(d_hiddens, d_hiddens) for _ in range(n_hiddens)]  # some hidden layers
        layers.append(nn.Linear(d_hiddens, output_size))
        # TODO: beware of last layer, often it isn't normalized
        #   AND one must take care of the output range to match that of the target data (e.g. regression task)
        self.classifier = nn.Sequential(*layers)
    def forward(self, x):
        return self.classifier(x)

model4 = MLP_classifier(n_features, n_classes, n_hidden_layers, hidden_size).to(device)
output4 = model4(input1)
print(f"\nmodel4: forwarded input1 of shape {input1.shape} to output4 of shape {output4.shape}")
# model4: forwarded input1 of shape torch.Size([8, 4]) to output4 of shape torch.Size([8, 3])
# TODO: here the output is some "weights" for each of the n_classes
#   you can convert logits to probas. with torch.softmax
print_trainable_parameters(model4, verbose=False)
# trainable params: 771 || all params: 771 || trainable%: 100.00

########################################################################################################################
## !!! Pitfalls !!!

class buggy_model(nn.Module):
    def __init__(self, input_size, output_size, d_hiddens):
        super(buggy_model, self).__init__()
        # correct way to add a list of modules (e.g. as opposed to a fixed ordering in Sequential)
        self.layers1 = nn.ModuleList([LinearBlock_A(input_size, d_hiddens), LinearBlock_A(d_hiddens, d_hiddens),
                                      nn.Linear(d_hiddens, output_size)])
        # FIXME: below, the plain list of nn.Module won't be registered to the state_dict and not trained with the optimizer
        self.layers2 = [LinearBlock_A(input_size, d_hiddens), LinearBlock_A(d_hiddens, d_hiddens),
                        nn.Linear(d_hiddens, output_size)]
        # FIXME: below, the torch.tensor won't be registered to the state_dict and not trained with the optimizer
        self.learned_scaling = torch.ones(input_size, dtype=torch.float32, requires_grad=True)
        self.fixed_scaling = torch.ones(input_size, dtype=torch.float32, requires_grad=False)
    def forward(self, x):
        for i, l in enumerate(self.layers1):
            x = l(x)
        return x

model5 = buggy_model(n_features, n_classes, hidden_size).to(device)
output5 = model5(input1)
print(f"\nmodel5: forwarded input1 of shape {input1.shape} to output5 of shape {output5.shape}")
# model5: forwarded input1 of shape torch.Size([8, 4]) to output5 of shape torch.Size([8, 3])
print(model5.state_dict().keys())  # only params from layers1
print_trainable_parameters(model5, verbose=False)
# trainable params: 467 || all params: 467 || trainable%: 100.00 (all from layers1 and trainable)

