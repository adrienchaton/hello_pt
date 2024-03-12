# here are some misc. details and utils to optimise your codes
# this isn't intended to be run as the other scripts 0/1/2

# further reads at:
# - https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# - https://pytorch.org/docs/stable/notes/randomness.html
# - https://pytorch.org/tutorials/beginner/ptcheat.html
# - https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/


torch.backends.cudnn.benchmark = True  # will let CuDNN optimize your computation depending on hardware,
# favour parameters with dimensions as multiple of 8 for maximising tensor core usage
# and adapt float precision to your needs and hardware (will be handled by pt-lightning in the part2), for TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.randn((100,), device="cuda")  # instantiate on GPU memory when possible, instead of transferring with .cuda()
torch.randn((100,)).to(device, non_blocking=True)  # or transfer asynchronously

torch.utils.data.DataLoader(dataset, num_workers=3, shuffle=True, drop_last=True, pin_memory=True)
# use multiple CPU workers and pin_memory if GPU doesn't go OOM, drop_last=True along with ...cudnn.benchmark=True

# within e.g. the training loop, keep all data/processing on GPU and asynchronous
# no print(), no .cpu(), no .numpy(), no .item() BUT .detach() should be used to e.g. log loss values through an epoch

# watch-out for non-differentiable operations which prevent backprop. (e.g. in model forward)
x_notdiff = torch.argmax(x_diff)  # instead, look e.g. for Gumbel-Softmax (here x_notdiff.grad_fn=None)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
# but avoid using torch.use_deterministic_algorithms(True) except if explicitly required for reproducibility

# do not use int to represent categorical data (when there is no relation/ranking), instead use
nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
torch.nn.functional.one_hot(int_labels, num_classes=num_classes)

# init your nn.Module on cpu and only transfer to GPU the whole model once it is instantiated
self.layer = nn.Linear(in_dim, out_dim).to(device)  # DON'T
model = MyModel().to(device)  # DO this instead

torch.optim.AdamW(model1.parameters(), weight_decay=0.01)  # use AdamW instead of Adam is using weight_decay

nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # after loss.backward() and before optimizer.step(), to stabilise training

# if OOM (GPU), use gradient accumulation, i.e. summing losses over several forward passes before calling .backward()
# if OOM (RAM), look into iterable datasets and sharding
# https://pytorch.org/data/main/torchdata.datapipes.iter.html
# https://pytorch.org/docs/stable/notes/multiprocessing.html

torch.distributed.get_rank() == 0  # can be used along with multiprocessing to only do an operation on the main process

# monitoring resources within python, use psutil (disk, RAM, CPU) or pynvml
# https://suzyahyah.github.io/code/pytorch/2024/01/25/GPUTrainingHacks.html

