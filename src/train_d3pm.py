import argparse
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import wandb

from models.d3pm import D3PM
from models.dit_llama import DiT_Llama
from models.vqvae import VQVAE, load_model_params

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D3PM on CIFAR10")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=4000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--wandb_project", type=str, default="udl-diffusion-semantics", help="WandB project name (if logging)")
    parser.add_argument("--wandb_entity", type=str, default="ml_exp")
    parser.add_argument("--vqvae_path", type=str, default="../vqvae-cifar10.pth", help="Path to pretrained VQ-VAE")
    parser.add_argument('--use-vae', action='store_true')
    parser.add_argument("--save_dir", type=str, default="contents", help="Directory to save samples")

    # D3PM / diffusion
    parser.add_argument("--num_classes", type=int, default=128)          # number of discrete categories per pixel / token
    # if not using vae this should be set to 8
    parser.add_argument("--n_T", type=int, default=1000)    # final timestep
    parser.add_argument("--hybrid_loss_coeff", type=float, default=0.0)

    # DiT
    parser.add_argument("--dit_dim", type=int, default=1024)

    # Sampling
    parser.add_argument("--sample_batch", type=int, default=16)
    parser.add_argument("--sample_stride", type=int, default=40)
    parser.add_argument("--sample_every", type=int, default=600)

    # VQ-VAE latent
    parser.add_argument("--latent_size", type=int, default=8)

    # VQ-VAE architecture
    parser.add_argument("--vq_n_embeds", type=int, default=128)
    parser.add_argument("--vq_hidden_dim", type=int, default=64)
    parser.add_argument("--vq_res_blocks", type=int, default=2)
    parser.add_argument("--vq_conv_blocks", type=int, default=2)

    # Optim / logging
    parser.add_argument("--grad_clip", type=float, default=5.0)
    # parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=2000)

    return parser.parse_args()

def load_models(args: argparse.Namespace) -> Tuple[D3PM, Optional[VQVAE]]:
    N = args.num_classes  # number of classes for discretized state per pixel
    d3pm = D3PM(
        DiT_Llama(1 if args.use_vae else 3, N, dim=args.dit_dim),
        args.n_T,
        num_classes=N,
        hybrid_loss_coeff=args.hybrid_loss_coeff,
    ).to(args.device)
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")

    # Load pretrained VQ-VAE
    vqvae = None
    if args.use_vae:
        vqvae = VQVAE(
            channels=3,
            n_embeds=args.vq_n_embeds,
            hidden_dim=args.vq_hidden_dim,
            n_pixelcnn_res_blocks=args.vq_res_blocks,
            n_pixelcnn_conv_blocks=args.vq_conv_blocks,
        ).to(args.device)

        # Load pretrained weights
        load_model_params(vqvae, args.vqvae_path, args.device, strict=True)
        vqvae.eval()  # Freeze VQ-VAE
        print(f"Loaded (pretrained) VAE, Total Param Count: {sum([p.numel() for p in vqvae.parameters()])}")
    else:
        print("No vae selected, will attempt to discritize input by scaling to nearest integer")
    
    return d3pm, vqvae

def load_cifar10_dataset(args: argparse.Namespace) -> DataLoader:
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

def training_iter(args: argparse.Namespace,
                  d3pm: D3PM, vqvae: VQVAE, dataloader: DataLoader, optim, global_step: int) -> int:
    pbar = tqdm(dataloader)
    loss_ema = None
    N = args.num_classes  # number of classes for discretized state per pixel
    for x, cond in pbar:
        optim.zero_grad()
        x = x.to(args.device)
        cond = cond.to(args.device)

        # discretize x to N bins
        if args.use_vae:
            with torch.no_grad():   # might not be necessary
                x_cat = vqvae.get_post_q(x)
                # print(x_cat.shape)
                x_cat = x_cat.unsqueeze(1)
                # print(x_cat.shape)
        else:
            x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)

        loss, info = d3pm(x_cat, cond)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), args.grad_clip)

        with torch.no_grad():
            param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

        if global_step % 10 == 0 and args.wandb_project:
            wandb.log(
                {
                    "train_loss": loss,
                    "train_grad_norm": norm,
                    "train_param_norm": param_norm,
                }
            )

        pbar.set_description(
            f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
        )
        optim.step()
        global_step += 1

        if global_step % args.sample_every == 1:
            eval_sample_image(args, d3pm, vqvae, x_cat, global_step)

        if global_step % args.ckpt_every == 1:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "step": global_step,
                    "model": d3pm.state_dict(),
                    "optimizer": optim.state_dict(),
                },
                f"checkpoints/d3pm_step_{global_step}.pth",
                )
    return global_step

def eval_sample_image(args: argparse.Namespace,
                      d3pm: D3PM, vqvae: VQVAE, x_cat: torch.Tensor, global_step: int) -> None:
    N = args.num_classes  # number of classes for discretized state per pixel
    d3pm.eval()

    with torch.no_grad():
        cond = torch.arange(0, args.sample_batch).to(args.device) % 10
        if args.use_vae:
            init_noise = torch.randint(0, N, (args.sample_batch, 1, args.latent_size, args.latent_size)).to(args.device)
        else:
            # These values are hardcoded to be that of the cifar10 dataset
            # This is inherited from the original script that this was based on which I won't change since
            # we are not explicitly using it.
            init_noise = torch.randint(0, N, (args.sample_batch, 3, 32, 32)).to(args.device)

        images = d3pm.sample_with_image_sequence(init_noise, cond, stride=args.sample_stride)

        # image sequences to gif
        gif = []
        if args.use_vae:
            q_sequence = images
            for q in q_sequence:
                q = q.squeeze(1)              # (B, 8, 8)
                img = vqvae.q_to_image(q)     # (B, 3, 32, 32)
                img = (img + 1) / 2            # [-1,1] → [0,1]

                grid = make_grid(img, nrow=4)
                np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gif.append(Image.fromarray(np_img))
        else:
            for image in images:
                x_from_dataloader = x_cat[:16].cpu() / (N - 1)
                this_image = image.float().cpu() / (N - 1)
                all_images = torch.cat([x_from_dataloader, this_image], dim=0)
                x_as_image = make_grid(all_images, nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

        gif[0].save(
            f"{args.save_dir}/sample_{global_step}.gif",
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )

        last_img = gif[-1]
        last_img.save(f"{args.save_dir}/sample_{global_step}_last.png")

        # log images
        if args.wandb_project:
            wandb.log(
                {
                    "sample": wandb.Image(last_img),
                    "global_step": global_step,
                }
            )
    d3pm.train()

if __name__ == "__main__":
    args = parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    os.makedirs(args.save_dir, exist_ok=True)
    
    # load/initiate models, dataset, optimizer
    d3pm, vqvae = load_models(args)    
    dataloader = load_cifar10_dataset(args)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=args.lr)
    d3pm.train()

    global_step = 0
    for i in range(args.epochs):
        global_step = training_iter(args, d3pm, vqvae, dataloader, optim, global_step)    # returns updated global_step

    # Save the model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(d3pm.state_dict(), "checkpoints/d3pm_final.pth")
    print("Model saved to checkpoints/d3pm_final.pth")
