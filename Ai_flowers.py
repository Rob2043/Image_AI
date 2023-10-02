import numpy as np
import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm


def main():
    random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    prepare_imgs = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ImageFolder("C:\Telegram bot\Dataset", transform=prepare_imgs)

    class ValueMeter(object):
        def __init__(self):
            self.sum = 0
            self.total = 0

        def add(self, value, n):
            self.sum += value * n
            self.total += n

        def value(self):
            return self.sum / self.total

    def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None):
        print(
            f"[{mode}] Epoch: {epoch:0.2f}. "
            f"Loss: {loss_meter.value():.2f}. "
            f"Accuracy: {100*accuracy_meter.value():.2f}% ",
            end="\n",
        )

        if best_perf:
            print(f"[best: {best_perf:0.2f}]%", end="")

    batch_size = 32
    lr = 0.001

    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - 1000, 1000]
    )
    print("Размер обучающего и валидационного датасета: ", len(train_set), len(val_set))
    loaders = {
        "training": DataLoader(
            train_set, batch_size, pin_memory=True, num_workers=2, shuffle=True
        ),
        "validation": DataLoader(
            val_set, batch_size, pin_memory=True, num_workers=2, shuffle=False
        ),
    }

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def trainval(model, loaders, optimizer, epochs=10):
        loss_meter = {"training": ValueMeter(), "validation": ValueMeter()}
        accuracy_meter = {"training": ValueMeter(), "validation": ValueMeter()}

        loss_track = {"training": [], "validation": []}
        accuracy_track = {"training": [], "validation": []}

        for epoch in range(epochs):
            for mode in ["training", "validation"]:
                with torch.set_grad_enabled(mode == "training"):
                    model.train() if mode == "training" else model.eval()
                    for imgs, labels in tqdm(loaders[mode]):
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        bs = labels.shape[0]

                        preds = model(imgs)
                        loss = F.cross_entropy(preds, labels)
                        acc = accuracy(preds, labels)

                        loss_meter[mode].add(loss.item(), bs)
                        accuracy_meter[mode].add(acc, bs)

                        if mode == "training":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                log(mode, epoch, loss_meter[mode], accuracy_meter[mode])

                loss_track[mode].append(loss_meter[mode].value())
                accuracy_track[mode].append(accuracy_meter[mode].value())
        return loss_track, accuracy_track

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    model = torchvision.models.mobilenet_v3_small(pretrained="imagenet", progress=True)

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    model.resnet_fc = nn.Linear(1024, 5)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    loss_track, accuracy_track = trainval(model, loaders, optimizer, epochs=10)

    # from matplotlib import pyplot as plt

    # plt.plot(accuracy_track["training"], label="train")
    # plt.plot(accuracy_track["validation"], label="val")
    # plt.ylabel("accuracy")
    # plt.xlabel("epoch")
    # plt.grid()
    # plt.legend()

    # plt.plot(loss_track["training"], label="train")
    # plt.plot(loss_track["validation"], label="val")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.grid()
    # plt.legend()

    # Load the modified layers from your saved state_dict
    saved_state_dict = torch.load("Ai_flowers.pth")
    model_state_dict = model.state_dict()

    # Update the model's state_dict with the saved state_dict for matching keys
    for key in model_state_dict.keys():
        if key in saved_state_dict and "resnet_fc" in key:
            model_state_dict[key] = saved_state_dict[key]
            
    torch.save(model.state_dict(), "updated_model.pth")


if __name__ == "__main__":
    main()
