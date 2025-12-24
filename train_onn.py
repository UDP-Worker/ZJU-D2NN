"""
Optimize the images loaded to SLM2 i.e. the weight of optical neural network
to recognize MNIST handwritten digits.

Loss is defined as the cross entropy loss, while the grey scale of certain area
on the camera is recognized as probability.
"""

import math
import utils
import torch
import argparse
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TrainONNSystem(nn.Module):
    def __init__(
            self,
            propagation1: nn.Module,  # SLM1 to SLM2
            propagation2: nn.Module,  # SLM2 to camera
            slm2_init: torch.Tensor,   # Initial phase pattern for SLM2
            slm2_learnable: bool = True,
    ):
        super().__init__()
        self.propagation1 = propagation1
        self.propagation2 = propagation2

        # SLM2 phase pattern
        if slm2_learnable:
            self.slm2_phase = nn.Parameter(slm2_init)
        else:
            self.register_buffer("slm2_phase", slm2_init, persistent=True)

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        input_field: (..., H, W) complex field at SLM1 plane
        return: (..., H, W) complex field at camera plane
        """
        # Propagate from SLM1 to SLM2
        field_slm2 = self.propagation1(input_field)  # (..., H, W)

        # Apply SLM2 phase modulation
        slm2_phase_exp = torch.exp(1j * self.slm2_phase)  # (H, W)
        field_slm2_modulated = field_slm2 * slm2_phase_exp  # (..., H, W)

        # Propagate from SLM2 to camera
        field_camera = self.propagation2(field_slm2_modulated)  # (..., H, W)

        return field_camera
    
    
def calculate_loss(
    field_camera: torch.Tensor,      # (B, H, W) complex
    labels: torch.Tensor,            # (B,) long, 0..9
    readout_pixels: torch.Tensor,    # (10, 2) long, each [y, x]
    eps: float = 1e-12,
) -> torch.Tensor:


    intensity = torch.abs(field_camera) ** 2  # (B, H, W)

    readout_pixels = readout_pixels.to(device=intensity.device, dtype=torch.long)
    ys = readout_pixels[:, 0]
    xs = readout_pixels[:, 1]

    scores = intensity[:, ys, xs]  # (B, 10), non-negative

    probs = scores / (scores.sum(dim=1, keepdim=True) + eps)  # (B, 10)
    log_probs = torch.log(probs + eps)

    labels = labels.to(device=intensity.device, dtype=torch.long)
    loss = F.nll_loss(log_probs, labels)

    return loss


def embed_mnist_to_field(images: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    将 MNIST (B,1,28,28) 的灰度图当作“幅度”，嵌入到 (H,W) 的中心；
    相位设为 0，得到 (B,H,W) 的复数光场（complex64）。
    """
    assert images.ndim == 4 and images.shape[1:] == (1, 28, 28)
    B = images.shape[0]

    amp = torch.zeros((B, H, W), device=images.device, dtype=torch.float32)

    y0 = (H - 28) // 2
    x0 = (W - 28) // 2
    amp[:, y0:y0 + 28, x0:x0 + 28] = images[:, 0, :, :]

    field = torch.complex(amp, torch.zeros_like(amp))
    return field.to(torch.complex64)


# --------- 读出像素（10类对应10个点）---------
def make_readout_pixels(H: int, W: int) -> torch.Tensor:
    """
    默认给 10 个类别指定 10 个像素点 (y,x)。
    你后续可以替换为真实标定得到的读出点/读出ROI中心。
    """
    cy = H // 2
    xs = torch.linspace(W * 0.2, W * 0.8, steps=10).round().long()
    ys = torch.full((10,), cy, dtype=torch.long)
    return torch.stack([ys, xs], dim=1)  # (10,2)


@torch.no_grad()
def evaluate(model, loader, readout_pixels, H, W, device):
    """
    测试集评估：返回 (平均loss, 准确率)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        input_field = embed_mnist_to_field(images, H, W)
        field_camera = model(input_field)

        loss = calculate_loss(field_camera, labels, readout_pixels)
        total_loss += float(loss.item()) * images.size(0)

        # 用读出点强度最大者作为预测类别
        intensity = torch.abs(field_camera) ** 2
        ys = readout_pixels[:, 0].to(device)
        xs = readout_pixels[:, 1].to(device)
        scores = intensity[:, ys, xs]  # (B,10)
        pred = scores.argmax(dim=1)

        total_correct += int((pred == labels).sum().item())
        total_n += images.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ============================
# propagation 模块加载接口
# ============================

def load_propagation_modules_from_torchscript(prop1_ts: str, prop2_ts: str, device: torch.device):
    """
    从 TorchScript 文件加载两个传播网络
    要求你提前把 propagation1/2 导出成 torchscript：torch.jit.save(...)
    """
    p1 = torch.jit.load(prop1_ts, map_location=device).eval()
    p2 = torch.jit.load(prop2_ts, map_location=device).eval()
    return p1, p2



def main():
    # -------------------------
    # 1) 命令行参数
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=128, help="仿真平面高度(像素)")
    parser.add_argument("--W", type=int, default=128, help="仿真平面宽度(像素)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST 数据目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="保存目录")
    parser.add_argument("--save_name", type=str, default="onn_slm2.pt", help="保存文件名")
    parser.add_argument("--no_cuda", action="store_true", help="强制不用GPU")

    # propagation：TorchScript 路径（方式A）
    parser.add_argument("--prop1_ts", type=str, default="", help="propagation1 的 TorchScript 文件路径")
    parser.add_argument("--prop2_ts", type=str, default="", help="propagation2 的 TorchScript 文件路径")

    # SLM2 初始化
    parser.add_argument("--slm2_init", type=str, default="", help="可选：加载 slm2_init 的 .pt 文件（形状(H,W)）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # -------------------------
    # 2) 设备选择
    # -------------------------
    device = torch.device("cuda" if (torch.cuda.is_available() and (not args.no_cuda)) else "cpu")
    print(f"[INFO] device = {device}")

    H, W = args.H, args.W
    if H < 28 or W < 28:
        raise ValueError("H 和 W 必须 >= 28，才能容纳 MNIST 28x28。")

    # -------------------------
    # 3) 数据加载（MNIST）
    # -------------------------
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------
    # 4) 加载 propagation1 / propagation2
    # -------------------------
    propagation1, propagation2 = load_propagation_modules_from_torchscript(
        args.prop1_ts, args.prop2_ts, device
    )
    print("[INFO] 已从 TorchScript 加载 propagation1/2")


    # -------------------------
    # 5) 初始化 slm2_phase
    # -------------------------
    if args.slm2_init:
        slm2_init = torch.load(args.slm2_init, map_location="cpu")
        if not (isinstance(slm2_init, torch.Tensor) and slm2_init.shape == (H, W)):
            raise ValueError(f"slm2_init 必须是形状 (H,W)=({H},{W}) 的 torch.Tensor")
        slm2_init = slm2_init.to(torch.float32)
        print(f"[INFO] 已加载 slm2_init: {args.slm2_init}")
    else:
        slm2_init = (2 * math.pi) * torch.rand((H, W), dtype=torch.float32)
        print("[INFO] 使用随机 slm2_init ∈ [0, 2π)")

    # -------------------------
    # 6) 构建 ONN 系统
    # -------------------------
    model = TrainONNSystem(
        propagation1=propagation1,
        propagation2=propagation2,
        slm2_init=slm2_init,
        slm2_learnable=True,
    ).to(device)

    # 冻结参数：
    for p in model.propagation1.parameters():
        p.requires_grad_(False)
    for p in model.propagation2.parameters():
        p.requires_grad_(False)

    readout_pixels = make_readout_pixels(H, W)

    optimizer = torch.optim.Adam([model.slm2_phase], lr=args.lr)

    # -------------------------
    # 7) 训练循环
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0

        for images, labels in train_loader:
            images = images.to(device)   # (B,1,28,28)
            labels = labels.to(device)   # (B,)

            input_field = embed_mnist_to_field(images, H, W)  # (B,H,W) complex
            field_camera = model(input_field)

            loss = calculate_loss(field_camera, labels, readout_pixels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            running_n += images.size(0)

        train_loss = running_loss / max(running_n, 1)
        test_loss, test_acc = evaluate(model, test_loader, readout_pixels, H, W, device)

        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc*100:.2f}%")

    # -------------------------
    # 8) 保存
    # -------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / args.save_name

    ckpt = {
        "slm2_phase": model.slm2_phase.detach().cpu(),        # 最终参数
        "model_state_dict": model.state_dict(),
        "H": H,
        "W": W,
        "readout_pixels": readout_pixels.cpu(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[INFO] 已保存到: {ckpt_path}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()