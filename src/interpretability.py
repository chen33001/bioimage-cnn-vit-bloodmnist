import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import INFO

from torch.utils.data import DataLoader

from src.data_loader import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_dataloaders, set_seed
from src.train_cnn import build_model as build_cnn
from src.train_vit import build_model as build_vit


matplotlib.use("Agg")


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Revert ImageNet normalisation back to 0-1 space."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(1, -1, 1, 1)
    return image * std + mean


def save_overlay(
    base_image: torch.Tensor,
    heatmap: np.ndarray,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    """Persist overlay between base image (tensor) and heatmap (0-1 np array)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = denormalize(base_image.unsqueeze(0)).squeeze().permute(1, 2, 0)
    image = image.clamp(0, 1).cpu().numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


class GradCAM:
    """Minimal Grad-CAM implementation for CNN backbones."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _, __, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _, grad_input, grad_output) -> None:  # type: ignore[override]
        # grad_output is a tuple; we only need the first element
        self.gradients = grad_output[0].detach()

    def generate(self, inputs: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Run forward/backward pass and return Grad-CAM map + logits.

        Parameters
        ----------
        inputs:
            Tensor of shape (1, C, H, W)
        class_idx:
            Target class index; default to predicted argmax.
        """
        if inputs.ndim != 4 or inputs.shape[0] != 1:
            raise ValueError("GradCAM expects single-sample batch of shape (1, C, H, W)")
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")
        weights = self.gradients.mean(dim=(-1, -2), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, logits

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _cnn_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    name = arch.lower()
    if name == "resnet18":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer defined for CNN arch '{arch}'")


class VitAttentionRollout:
    """
    Attention rollout for ViT-style models with class token.
    Captures QKV projections to reconstruct attention weights, ensuring compatibility with timm models.
    """

    def __init__(self, model: torch.nn.Module, image_size: int):
        if not hasattr(model, "blocks"):
            raise ValueError("Attention rollout currently supports VisionTransformer backbones with .blocks")
        self.model = model
        self.image_size = image_size
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._qkv_cache: List[Tuple[int, torch.Tensor]] = []
        for block in model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "qkv"):
                continue
            handle = attn.qkv.register_forward_hook(self._make_qkv_hook(attn.num_heads))
            self.handles.append(handle)

    def _make_qkv_hook(self, num_heads: int):
        def hook(_module, _inputs, output):
            self._qkv_cache.append((num_heads, output.detach()))

        return hook

    def _compute_attention_matrices(self, device: torch.device) -> torch.Tensor:
        if not self._qkv_cache:
            raise RuntimeError("No attention matrices captured; ensure model uses standard ViT blocks.")
        attn_maps: List[torch.Tensor] = []
        for num_heads, qkv in self._qkv_cache:
            B, N, threeC = qkv.shape
            head_dim = threeC // (3 * num_heads)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]  # each shape: (B, num_heads, N, head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = torch.softmax(attn, dim=-1)
            attn_maps.append(attn.mean(dim=1))  # average over heads -> (B, N, N)
        stacked = torch.stack(attn_maps)  # (layers, B, N, N)
        return stacked.squeeze(1).to(device)  # remove batch dim (B==1)

    def generate(self, inputs: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[0] != 1:
            raise ValueError("Attention rollout expects single-sample batch of shape (1, C, H, W)")
        self._qkv_cache.clear()
        logits = self.model(inputs)
        attn_stack = self._compute_attention_matrices(inputs.device)
        tokens = attn_stack.size(-1)
        identity = torch.eye(tokens, device=attn_stack.device)
        rollout = attn_stack + identity
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        joint = rollout[0]
        for i in range(1, rollout.size(0)):
            joint = rollout[i] @ joint
        mask = joint[0, 1:]  # attention from CLS token to patch tokens
        num_patches = mask.size(0)
        grid_size = int(math.sqrt(num_patches))
        mask = mask.reshape(1, 1, grid_size, grid_size)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask, logits

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def collect_samples(
    loader: DataLoader,
    num_classes: int,
    samples_per_class: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Yield up to N samples per class from loader."""
    counts: Dict[int, int] = defaultdict(int)
    total_needed = num_classes * samples_per_class
    yielded = 0
    index = 0
    for images, labels in loader:
        for i in range(images.size(0)):
            label = int(labels[i].item())
            if counts[label] >= samples_per_class:
                index += 1
                continue
            counts[label] += 1
            yielded += 1
            yield images[i : i + 1], labels[i : i + 1], index
            if yielded >= total_needed:
                return
            index += 1


def generate_cnn_overlays(
    arch: str,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    samples_per_class: int,
    data_dir: str,
    seed: int,
    device: torch.device,
    image_size: int,
) -> None:
    set_seed(seed)
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        image_size=image_size,
        augment=False,
        download=True,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    class_names = list(INFO["bloodmnist"]["label"].values())
    model = build_cnn(arch, n_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model, _cnn_target_layer(model, arch))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for img, label, idx in collect_samples(loader, n_classes, samples_per_class):
            img = img.to(device)
            cam, logits = gradcam.generate(img)
            pred = int(torch.argmax(logits, dim=1).item())
            title = f"Label: {class_names[label.item()]} | Pred: {class_names[pred]}"
            out_path = output_dir / f"{arch}_{split}_idx{idx:05d}_label{label.item()}_pred{pred}.png"
            save_overlay(img.squeeze().cpu(), cam, out_path, title)
    finally:
        gradcam.close()


def generate_vit_overlays(
    arch: str,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    samples_per_class: int,
    data_dir: str,
    seed: int,
    device: torch.device,
    image_size: int,
) -> None:
    set_seed(seed)
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        image_size=image_size,
        augment=False,
        download=True,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    class_names = list(INFO["bloodmnist"]["label"].values())

    model = build_vit(arch, n_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model = model.to(device)
    model.eval()

    rollout = VitAttentionRollout(model, image_size=image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for img, label, idx in collect_samples(loader, n_classes, samples_per_class):
            img = img.to(device)
            attn_map, logits = rollout.generate(img)
            pred = int(torch.argmax(logits, dim=1).item())
            title = f"Label: {class_names[label.item()]} | Pred: {class_names[pred]}"
            out_path = output_dir / f"{arch}_{split}_idx{idx:05d}_label{label.item()}_pred{pred}.png"
            save_overlay(img.squeeze().cpu(), attn_map, out_path, title)
    finally:
        rollout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interpretability overlays for BloodMNIST models.")
    parser.add_argument("--model-type", choices=["cnn", "vit"], required=True)
    parser.add_argument("--arch", required=True, help="Model architecture string (e.g., resnet18, vit_base_patch16_224)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "interpretability")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default="auto", help="Device to run on: auto|cpu|cuda|cuda:0")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.model_type == "cnn":
        generate_cnn_overlays(
            arch=args.arch,
            checkpoint=args.checkpoint,
            split=args.split,
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            data_dir=args.data_dir,
            seed=args.seed,
            device=device,
            image_size=args.image_size,
        )
    else:
        generate_vit_overlays(
            arch=args.arch,
            checkpoint=args.checkpoint,
            split=args.split,
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            data_dir=args.data_dir,
            seed=args.seed,
            device=device,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
