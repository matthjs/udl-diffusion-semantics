import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from torchvision.models import resnet18, ResNet18_Weights

from src.utils.utils import get_device, set_seed, save_model_params
from src.data.cifar10 import get_dls


def get_args(to_upperse=True):
    p = argparse.ArgumentParser()

    # Data / train
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_cpus", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Fine-tuning
    p.add_argument("--freeze_until", type=str, default="layer4",
                   choices=["none", "layer3", "layer4"],
                   help="Which layers to keep trainable initially.")
    p.add_argument("--unfreeze_epoch", type=int, default=0,
                   help="If >0, at this epoch unfreeze everything (except maybe BN if you want).")

    args = p.parse_args()
    if to_upperse:
        d = vars(args)
        args = argparse.Namespace(**{k.upper(): v for k, v in d.items()})
    return args


def build_resnet18_cifar10(pretrained=True, n_classes=10):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def set_trainable(model, freeze_until: str):
    """
    freeze_until:
      - "layer4": train only layer4 + fc
      - "layer3": train layer3 + layer4 + fc
      - "none": train all
    """
    for p in model.parameters():
        p.requires_grad = True

    if freeze_until == "none":
        return

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze selected
    if freeze_until == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    elif freeze_until == "layer3":
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True


@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    total = 0
    correct = 0
    cum_loss = 0.0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        cum_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return cum_loss / max(total, 1), correct / max(total, 1)


def main():
    args = get_args()
    set_seed(args.SEED)
    device = get_device()

    train_dl, val_dl, test_dl = get_dls(
        data_dir=args.DATA_DIR,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        val_ratio=args.VAL_RATIO,
        seed=args.SEED,
    )

    save_dir = Path(args.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18_cifar10(pretrained=True, n_classes=10).to(device)

    # initial freeze strategy
    set_trainable(model, args.FREEZE_UNTIL)

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optim, T_max=args.N_EPOCHS)

    best_val_acc = -math.inf
    best_path = None

    for epoch in range(1, args.N_EPOCHS + 1):
        # optional unfreeze
        if args.UNFREEZE_EPOCH and epoch == args.UNFREEZE_EPOCH:
            set_trainable(model, "none")
            optim = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.LR * 0.3,
                weight_decay=args.WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(optim, T_max=args.N_EPOCHS - epoch + 1)
            print(f"[UNFREEZE] epoch={epoch}: now training all layers.")

        model.train()
        total = 0
        correct = 0
        cum_loss = 0.0

        for x, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            cum_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = cum_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = eval_epoch(model, val_dl, device)
        print(
            f"[{epoch}/{args.N_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = save_dir / f"best_resnet18_rgb-epoch={epoch}-val_acc={val_acc:.4f}.pth"
            save_model_params(model=model, save_path=best_path)

        scheduler.step()

    if best_path is not None and test_dl is not None:
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        test_loss, test_acc = eval_epoch(model, test_dl, device)
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
        print(f"[BEST] {best_path}")


if __name__ == "__main__":
    main()