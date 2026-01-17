import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from src.utils.utils import get_device, set_seed
from src.data.cifar10 import get_dls


def get_args(to_upperse=True):
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_cpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--val_ratio", type=float, default=0.2)

    args = p.parse_args()
    if to_upperse:
        d = vars(args)
        args = argparse.Namespace(**{k.upper(): v for k, v in d.items()})
    return args


def build_resnet18_cifar10(pretrained=False, n_classes=10):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # CIFAR-10 adaptation
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Class head
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


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

    _, _, test_dl = get_dls(
        data_dir=args.DATA_DIR,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        val_ratio=args.VAL_RATIO,
        seed=args.SEED,
    )

    model = build_resnet18_cifar10(pretrained=False, n_classes=10).to(device)

    state = torch.load(args.CKPT, map_location=device)
    model.load_state_dict(state)
    print(f"[LOADED] {args.CKPT}")

    test_loss, test_acc = eval_epoch(model, test_dl, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()