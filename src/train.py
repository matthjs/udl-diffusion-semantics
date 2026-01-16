import argparse
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    os.makedirs(args.save_dir, exist_ok=True)
    N = 8  # number of classes for discretized state per pixel
    d3pm = D3PM(
        DiT_Llama(3, N, dim=1024), 1000, num_classes=N, hybrid_loss_coeff=0.0
    ).to(args.device)
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")

    # Load pretrained VQ-VAE
    vqvae = None
    if args.use_vae:
        vqvae = VQVAE(
            channels=3,
            n_embeds=128,
            hidden_dim=64,
            n_pixelcnn_res_blocks=2,
            n_pixelcnn_conv_blocks=2,
        ).to(args.device)
        
        # Load pretrained weights
        load_model_params(vqvae, args.vqvae_path, args.device, strict=True)
        vqvae.eval()  # Freeze VQ-VAE
    else:
        print("No vae selected, will attempt to discritize input by scaling to nearest integer")

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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=args.lr)
    d3pm.train()

    global_step = 0
    for i in range(args.epochs):

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(args.device)
            cond = cond.to(args.device)

            # discretize x to N bins
            if args.use_vae:
                with torch.no_grad():   # might not be necessary
                    x_cat = vqvae.get_post_q(x)
            else:
                x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)

            loss, info = d3pm(x_cat, cond)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)

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

            if global_step % 600 == 1:
                d3pm.eval()

                with torch.no_grad():
                    cond = torch.arange(0, 16).to(args.device) % 10
                    init_noise = torch.randint(0, N, (16, 3, 32, 32)).to(args.device)

                    images = d3pm.sample_with_image_sequence(
                        init_noise, cond, stride=40
                    )
                    # image sequences to gif
                    gif = []
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

    # Save the model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(d3pm.state_dict(), "checkpoints/d3pm_final.pth")
    print("Model saved to checkpoints/d3pm_final.pth")
