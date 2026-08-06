#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from resnet56_noskip import ResNet


# In[2]:


torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# In[5]:


# Perform tranformations
train_transform=transforms.Compose([
    transforms.Pad(4), 
    transforms.RandomCrop(32), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),   # CIFAR-10 mean 
        (0.2470, 0.2435, 0.2616))  # CIFAR-10 std
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),   # CIFAR-10 mean 
        (0.2470, 0.2435, 0.2616))  # CIFAR-10 std
])


# In[6]:


# Load Dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)


# In[7]:

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# In[10]:





# In[ ]:





# In[3]:


model=ResNet(num_classes=10,n=9).to(device) #Here we will use n=9 for 56 layer ResNet
#print(model) 

# verifying mps
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
print("Device:", device)
print("Model device:", next(model.parameters()).device)


# In[12]:


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,          # typical starting LR for ResNet on CIFAR-10
    momentum=0.9,    # Changing momentum to match Li et al to reproduce-like Figure 1(a)
    nesterov=True,
    weight_decay=5e-4
)

# divide lr by 10 at epochs corresponding to 32k and 48k iterations
epoch_32k = int(32000 / len(train_loader))
epoch_48k = int(48000 / len(train_loader))


# NOTE: The schduler below is for the learning rate set up for original ResNet paper
#scheduler = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer,
#    milestones=[epoch_32k, epoch_48k],
#    gamma=0.1
#)

#NOTE:  The schduler below is for the learning rate set up for Li et al (Figure 1(a))
scheduler = MultiStepLR(
    optimizer,
    milestones=[150, 225, 275],
    gamma=0.1
)


# In[ ]:


from tqdm import tqdm
if __name__ == '__main__':
    epochs = 300

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    import os

    os.makedirs("checkpoints/sgd_nesterov", exist_ok=True)
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            unit="batch"
        )

        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_loss = total_loss / max(1, batch_bar.n + 1)
            current_acc = correct / total

            batch_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct_val = 0
        total_val = 0
    

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                _, predicted = torch.max(output, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        # Verify that the full CIFAR-10 test set was evaluated
        assert total_val == len(test_dataset), (
            f"Epoch {epoch + 1}: evaluated "
            f"{total_val}/{len(test_dataset)} test samples"
        )

        # Print raw validation counts every epoch
        print(
            f"Epoch {epoch + 1:3d} | "
            f"train={train_acc:.4f} | "
            f"val={val_acc:.4f} | "
            f"correct={correct_val}/{total_val}"
        )

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)


        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
                "seed": 42,
                "optimizer": "SGD+Nesterov",
                "rng_state": torch.get_rng_state(),
            }, f"checkpoints/sgd_nesterov/resnet56_epoch_{epoch+1}.pth")
        percent_done = 100 * (epoch + 1) / epochs

        epoch_bar.set_postfix({
            "done": f"{percent_done:.1f}%",
            "loss": f"{avg_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "lr": scheduler.get_last_lr()[0]
        })

    history = {
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies,
    }

    import json

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
# In[17]:


# Define filename (saves in current directory)
if __name__ == '__main__':
    model_path = 'sgdmomentum_resnet56_noskip_cifar10_like_fig1a_Li_et_al.pth'
    
    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epochs": epochs,
    }, model_path)


# In[19]:


if __name__ == '__main__':
    plt.figure(figsize=(10,5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('ResNet-56 Training on CIFAR-10')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[20]:


   
if __name__ == "__main__":
    classes = test_dataset.classes

    # Pick 5 random samples from the test set
    indices = random.sample(range(len(test_dataset)), 5)
    images, labels = zip(*[test_dataset[i] for i in indices])

    images_tensor = torch.stack(images).to(device)
    labels_tensor = torch.tensor(labels).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images_tensor)
        _, preds = torch.max(outputs, 1)

    def unnormalize(img):
        mean = torch.tensor(
            [0.4914, 0.4822, 0.4465]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.2470, 0.2435, 0.2616]
        ).view(3, 1, 1)

        return img.cpu() * std + mean

    plt.figure(figsize=(12, 4))

    for i in range(5):
        img = unnormalize(images_tensor[i])
        npimg = np.transpose(img.numpy(), (1, 2, 0))

        plt.subplot(1, 5, i + 1)
        plt.imshow(np.clip(npimg, 0, 1))
        plt.title(
            f"Pred: {classes[preds[i].item()]}\n"
            f"GT: {classes[labels[i]]}"
        )
        plt.axis("off")

    plt.suptitle(
        "ResNet-56 Predictions vs Ground Truth (CIFAR-10)"
    )
    plt.tight_layout()
    plt.show()
