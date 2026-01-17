import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.utils import get_device, set_seed, load_model_params, save_model_params
from src.models.vqvae import VQVAE
from src.models.classifier import QClassifierCNN


def get_args(to_upperse=True):
    p = argparse.ArgumentParser()

    # Training / data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--n_epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--n_cpus", type=int, required=True)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Pretrained VQVAE (feature extractor)
    p.add_argument("--vqvae_params", type=str, required=True)

    # VQVAE architecture MUST match checkpoint
    p.add_argument("--n_embeds", type=int, required=True)  # K
    p.add_argument("--hidden_dim", type=int, required=True)  # D
    p.add_argument("--n_pixelcnn_res_blocks", type=int, required=True)
    p.add_argument("--n_pixelcnn_conv_blocks", type=int, required=True)

    # Classifier options
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--freeze_codebook", action="store_true")
    p.add_argument("--init_from_vqvae_codebook", action="store_true")

    p.add_argument("--lambda_cls", type=float, default=0.1)   # weight of classification loss
    p.add_argument("--commit_weight", type=float, default=0.25)
    p.add_argument("--finetune_codebook", action="store_true") 

    args = p.parse_args()

    if to_upperse:
        d = vars(args)
        args = argparse.Namespace(**{k.upper(): v for k, v in d.items()})
    return args


@torch.no_grad()
def eval_epoch(vqvae, clf, dl, device):
    vqvae.eval()
    clf.eval()

    total = 0
    correct = 0
    cum_loss = 0.0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        q = vqvae.get_post_q(x)          # (B, 8, 8) for CIFAR10
        logits = clf(q)                  # (B, 10)
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

    from src.data.cifar10 import get_dls
    channels = 3
    n_classes = 10

    train_dl, val_dl, test_dl = get_dls(
        data_dir=args.DATA_DIR,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        val_ratio=args.VAL_RATIO,
        seed=args.SEED,
    )

    save_dir = Path(args.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained VQVAE
    vqvae = VQVAE(
        channels=channels,
        n_embeds=args.N_EMBEDS,
        hidden_dim=args.HIDDEN_DIM,
        n_pixelcnn_res_blocks=args.N_PIXELCNN_RES_BLOCKS,
        n_pixelcnn_conv_blocks=args.N_PIXELCNN_CONV_BLOCKS,
    ).to(device)

    load_model_params(model=vqvae, model_params=args.VQVAE_PARAMS, device=device, strict=True)

    # Freeze decoder + PixelCNN (we don't need them for the classifier objective)
    for p in vqvae.dec.parameters():
        p.requires_grad = False
    for p in vqvae.pixelcnn.parameters():
        p.requires_grad = False

    # Encoder is trainable
    for p in vqvae.enc.parameters():
        p.requires_grad = True

    # Codebook: optional finetune
    for p in vqvae.vect_quant.parameters():
        p.requires_grad = bool(args.FINETUNE_CODEBOOK)

    # Build classifier
    clf = QClassifierCNN(
        n_embeds=args.N_EMBEDS,
        hidden_dim=args.HIDDEN_DIM,
        n_classes=n_classes,
        dropout=args.DROPOUT,
        width=512,
        depth=13,
    ).to(device)

    # Optional: init classifier token embedding from VQ-VAE codebook
    if args.INIT_FROM_VQVAE_CODEBOOK:
        with torch.no_grad():
            clf.token_embed.weight.copy_(vqvae.vect_quant.embed_space.weight)

    if args.FREEZE_CODEBOOK:
        clf.token_embed.weight.requires_grad = False

    train_params = []
    train_params += [p for p in clf.parameters() if p.requires_grad]
    train_params += [p for p in vqvae.enc.parameters() if p.requires_grad]
    train_params += [p for p in vqvae.vect_quant.parameters() if p.requires_grad]

    optim = AdamW(train_params, lr=args.LR, weight_decay=2e-4)

    best_val_acc = -math.inf
    best_path = None

    for epoch in range(1, args.N_EPOCHS + 1):
        vqvae.train()
        clf.train()

        total = 0
        correct = 0
        cum_loss = 0.0

        for x, y in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            q = vqvae.get_post_q(x)  # (B, H, W) int64

            logits = clf(q)
            ce_loss = F.cross_entropy(logits, y)

            # keep recon objective to avoid destroying the representation
            vqvae_loss = vqvae.get_vqvae_loss(x, commit_weight=args.COMMIT_WEIGHT)

            loss = vqvae_loss + args.LAMBDA_CLS * ce_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            cum_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = cum_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = eval_epoch(vqvae, clf, val_dl, device)
        print(
            f"[{epoch}/{args.N_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = save_dir / f"best_q_classifier-epoch={epoch}-val_acc={val_acc:.4f}.pth"
            save_model_params(model=clf, save_path=best_path)

    # Final test (best checkpoint)
    if best_path is not None and test_dl is not None:
        load_model_params(model=clf, model_params=str(best_path), device=device, strict=True)
        test_loss, test_acc = eval_epoch(vqvae, clf, test_dl, device)
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
        print(f"[BEST] {best_path}")


if __name__ == "__main__":
    main()