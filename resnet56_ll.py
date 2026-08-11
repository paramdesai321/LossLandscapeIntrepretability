#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
https://www.kaggle.com/code/rohankt/resnet-implementation-from-scratch-on-cifar10/output
^ download link
'''


# In[3]:
import os
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resnet56_noskip import ResNet
from mps_configure import get_device


class Cifar10Batches(Dataset):
    def __init__(self, root, train=False, normalize=True):

        self.root = root
        self.train = train
        self.normalize = normalize

        files = []
        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        data_list, labels_list = [], []
        for fname in files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")

            data_list.append(entry[b"data"])
            labels_list.extend(entry[b"labels"])

        data = np.concatenate(data_list, axis=0)
        self.labels = np.array(labels_list, dtype=np.int64)

        data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.data = data

        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
        self.std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3,1,1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.normalize:
            x = (x - self.mean) / self.std
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)



# In[12]:


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            ch_pad = planes - in_planes
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::stride, ::stride],
                    (0, 0, 0, 0, 0, ch_pad),  # pad channels
                    mode="constant",
                    value=0.0
                )
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x)) # with skip connections
        out = self.relu(out)
        return out


class BasicBlockNoSkip(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        # No shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)   # <-- no residual addition
        return out


# In[13]:


def resnet56_cifar(num_classes=10):
    return CifarResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet56_cifar_noskip(num_classes=10):
    return CifarResNet(BasicBlockNoSkip, [9, 9, 9], num_classes)


# In[14]:


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = {k.replace("module.", ""): v for k, v in state.items()}


    remapped = {}
    for k, v in state.items():
        k2 = k

        if k2.startswith("bn."):
            k2 = "bn1." + k2[len("bn."):]

        if k2.startswith("stage1."):
            k2 = "layer1." + k2[len("stage1."):]
        elif k2.startswith("stage2."):
            k2 = "layer2." + k2[len("stage2."):]
        elif k2.startswith("stage3."):
            k2 = "layer3." + k2[len("stage3."):]

        remapped[k2] = v


    remapped = {k: v for k, v in remapped.items() if not k.endswith("num_batches_tracked")}

    # load
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print("Loaded checkpoint with strict=False")
    print("Missing keys (showing up to 10):", missing[:10])
    print("Unexpected keys (showing up to 10):", unexpected[:10])

    model.eval()
    return model


# In[15]:


def get_params_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_params_vector(model, vec):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset:offset+numel].view_as(p))
        offset += numel

#def random_direction_like(model, device): ### Method from the older version - to make plots like Goodfellow et. al
#    v = torch.cat([torch.randn_like(p, device=device).flatten() for p in model.parameters()])
#    v = v / (v.norm() + 1e-12)
#    return v


def random_direction_filterwise(model, device): # random_direction with Filter Normalization as done in Li et al.
    directions = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Ignore bias and BN params (paper uses xignore biasbn)
        if p.ndim == 1 or "bn" in name.lower() or "bias" in name.lower():
            directions.append(torch.zeros_like(p, device=device).flatten())
            continue

        # Random direction
        d = torch.randn_like(p, device=device)

        # Filter-wise normalization
        # Treat dim 0 as filter dimension
        d_view = d.view(d.shape[0], -1)
        w_view = p.detach().view(p.shape[0], -1)

        d_norm = d_view.norm(dim=1, keepdim=True) + 1e-12
        w_norm = w_view.norm(dim=1, keepdim=True)

        d_scaled = d_view / d_norm * w_norm
        directions.append(d_scaled.view_as(p).flatten())

    return torch.cat(directions)


# In[16]:


@torch.no_grad()
def eval_loss(model, loader, criterion, device, max_batches=10):
    model.eval()
    total_loss = 0.0
    total_n = 0

    for bi, (x, y) in enumerate(loader):

        if max_batches is not None and bi >= max_batches:
            break

        # move to device
        x = x.to(device)
        y = y.to(device)


        # extra safety check
        if x.dim() != 4:
            raise RuntimeError(f"Bad input shape for CNN: {x.shape}")

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


# In[17]:


def random_direction_weight_scaled(model, device):
    chunks = []
    for p in model.parameters():
        r = torch.randn_like(p, device=device)
        r = r * (p.detach().norm() / (r.norm() + 1e-12) + 1e-12)
        chunks.append(r.flatten())
    return torch.cat(chunks)




# In[ ]:





# In[18]:


import torch
import torch.nn as nn

@torch.no_grad()
def bn_reestimate(model, loader, device, num_batches=20):
    momenta = {}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            momenta[m] = m.momentum
            m.momentum = None  # cumulative moving average
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            if hasattr(m, "num_batches_tracked"):
                m.num_batches_tracked.zero_()

    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device, non_blocking=True)
        model(x)

    for m, mom in momenta.items():
        m.momentum = mom



# In[20]:


def main():
    # EDIT these if not in directory 
    ckpt_path = "" 
    cifar_batches_root = r"cifar-10-batches-py"
    assert os.path.isdir(cifar_batches_root), (
    f"Dataset directory not found: {cifar_batches_root}"
    )

    
    device = get_device()
    trainset = Cifar10Batches(cifar_batches_root, train=True, normalize=True)
    testset = Cifar10Batches(cifar_batches_root, train=False, normalize=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    bnloader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)

    #model = resnet56_cifar(num_classes=10).to(device)
    model = ResNet(num_classes=10, n=9).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Train accuracy: {checkpoint['train_acc']:.4f}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Model device: {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss()
    original_state = copy.deepcopy(model.state_dict())
    w0 = get_params_vector(model).clone()
    d1 = random_direction_filterwise(model, device)
    d2 = random_direction_filterwise(model, device)
    proj = torch.dot(d2, d1) / (torch.dot(d1, d1) + 1e-12)
    d2 = d2 - proj * d1

    xmin, xmax, xsteps = -1.0, 1.0, 21  # increasing the resolution
    ymin, ymax, ysteps = -1.0, 1.0, 21 # increasing the resolultion
    xs = np.linspace(xmin, xmax, xsteps)
    ys = np.linspace(ymin, ymax, ysteps)

    Z = np.zeros((ysteps, xsteps), dtype=np.float32)

    for i, beta in enumerate(ys):
        for j, alpha in enumerate(xs):
            w = w0 + alpha * d1 + beta * d2
            set_params_vector(model, w)
            bn_reestimate(model, bnloader, device, num_batches=5) 
            Z[i, j] = eval_loss(model, testloader, criterion, device, max_batches=20)
        print(f"row {i+1}/{ysteps} done")
    
    model.load_state_dict(original_state, strict=True)
    model.eval()

    print("Original model state restored.")
    


    print("Z min/max:", Z.min(), Z.max(), "range:", float(Z.max() - Z.min()))

    np.save("Z.npy", Z)
    np.save("xs.npy", xs)
    np.save("ys.npy", ys)
    print("Saved: Z.npy, xs.npy, ys.npy")

    from scipy.ndimage import gaussian_filter

    # we don't need smoothness via gaussian - Z_log is sufficient to reproduce Loss Landscape from Li et al.
    # Z_log = np.log(Z_clip - Z_clip.min() + 1e-6) # this will create artifical spike in minima
    Z_log = np.log(Z + 1e-8) # to omit the spike
    

    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    

    surf = ax.plot_surface(
        X, Y, Z_log, # replacing Z_smooth by Z_log here
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        shade=True
    )

    ax.set_xlabel("alpha (dir 1)")
    ax.set_ylabel("beta (dir 2)")
    ax.set_zlabel("log(loss) (smoothed)")
    ax.set_title("Loss Landscape (ResNet-56 CIFAR-10)")

    ax.view_init(elev=25, azim=-60)
    ax.set_zlim(Z_log.min(), np.percentile(Z_log, 98))

    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.savefig("loss_landscape_smooth.png", dpi=300)
    plt.show()

    print("Saved loss_landscape_smooth.png")

    return model, trainset, testset, device, Z, xs, ys

# in notebook do this so variables persist:
#model, trainset, testset, device, Z, xs, ys = main()

def setup(ckpt_path):

    cifar_batches_root = "./data/cifar-10-batches-py"

    device = get_device()

    trainset = Cifar10Batches(
        cifar_batches_root,
        train=True,
        normalize=True
    )

    testset = Cifar10Batches(
        cifar_batches_root,
        train=False,
        normalize=True
    )

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    testloader = DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    model = ResNet(
        num_classes=10,
        n=9
    ).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Model device: {next(model.parameters()).device}")

    return (
        model,
        trainset,
        testset,
        trainloader,
        testloader,
        device
    )



#model, trainset, testset, trainloader, testloader, device = setup()


# In[ ]:



# In[2]:


from torchvision.datasets import CIFAR10
CIFAR10(root="./data", train=True, download=True)


# In[8]:
def plot_1d_loss_landscape(
    model,
    bn_loader,
    eval_loader,
    criterion,
    device,
    direction=None,
    xmin=-1.0,
    xmax=1.0,
    steps=21,
    bn_batches=5,
    eval_batches=10,
    save_path=None,
    title="1D Loss Landscape",
):
    model.eval()

    # Save full original state, including BatchNorm buffers
    original_state = copy.deepcopy(model.state_dict())

    # Center = current checkpoint weights
    w_center = get_params_vector(model).clone()

    # Create direction if one wasn't supplied
    if direction is None:
        direction = random_direction_filterwise(
            model,
            device
        )

    xs = np.linspace(
        xmin,
        xmax,
        steps
    )

    losses = np.zeros(
        steps,
        dtype=np.float32
    )

    for i, alpha in enumerate(xs):

        w = w_center + alpha * direction

        set_params_vector(
            model,
            w
        )

        # Re-estimate BN using training data
        bn_reestimate(
            model,
            bn_loader,
            device,
            num_batches=bn_batches
        )

        # Evaluate desired split
        losses[i] = eval_loss(
            model,
            eval_loader,
            criterion,
            device,
            max_batches=eval_batches
        )

        print(
            f"[{i+1:02d}/{steps}] "
            f"alpha={alpha:+.3f} "
            f"loss={losses[i]:.4f}"
        )

    # Restore exact checkpoint state
    model.load_state_dict(
        original_state,
        strict=True
    )

    model.eval()

    print("Original model state restored.")

    # -----------------------------
    # Plot
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        xs,
        losses,
        marker="o"
    )

    plt.axvline(
        0.0,
        linestyle="--",
        label="Checkpoint"
    )

    plt.xlabel("alpha")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)

    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Save PNG if requested
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved plot: {save_path}")

    # Important: no GUI popup
    plt.close()

    return xs, losses, direction
# In[7]:


def plot_2d_loss_landscape(
    model,
    loader,
    criterion,
    device,
    x_range=(-1, 1),
    y_range=(-1, 1),
    steps=51
):
    model.eval()

    w0 = get_params_vector(model).clone()

    d1 = random_direction_filterwise(model, device)
    d2 = random_direction_filterwise(model, device)

    proj = torch.dot(d1, d2) / (torch.dot(d1, d1) + 1e-12)
    d2 = d2 - proj * d1

    xs = np.linspace(x_range[0], x_range[1], steps)
    ys = np.linspace(y_range[0], y_range[1], steps)

    Z = np.zeros((steps, steps), dtype=np.float32)

    for i, beta in enumerate(ys):
        for j, alpha in enumerate(xs):

            w = w0 + alpha * d1 + beta * d2
            set_params_vector(model, w)

            bn_reestimate(model, loader, device, num_batches=10)
            Z[i, j] = eval_loss(model, loader, criterion, device)

        print(f"row {i+1}/{steps} done")

    set_params_vector(model, w0)

    Z_log = np.log(Z + 1e-8)

    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z_log,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("log(loss)")
    ax.set_title("2D Loss Landscape")

    fig.colorbar(surf, shrink=0.6)
    plt.show()

    return Z, xs, ys    


# In[ ]:


from captum.attr import LayerGradCam, LayerAttribution
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_gradcam(model, image, label, device, target_layer):
    model.eval()

    image = image.unsqueeze(0).to(device)   # [1, 3, 32, 32]

    gradcam = LayerGradCam(model, target_layer)

    attributions = gradcam.attribute(
        image,
        target=label
    )

    # Upsample to input image size
    attributions = LayerAttribution.interpolate(
        attributions,
        image.shape[2:]
    )

    heatmap = attributions.squeeze().detach().cpu().numpy()
    img = image.squeeze().detach().cpu().numpy()

    # Handle channel dimension
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=0)

    # Normalize for visualization
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min() + 1e-8
    )

    # Convert CHW -> HWC
    img = img.transpose(1, 2, 0)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()

