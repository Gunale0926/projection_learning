from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(name: str, root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "MNIST":
        train = datasets.MNIST(root=root, train=True, download=False)
        test = datasets.MNIST(root=root, train=False, download=False)
        x_train = train.data.numpy().astype("float32")[:, None, :, :] / 255.0
        x_test = test.data.numpy().astype("float32")[:, None, :, :] / 255.0
        return x_train, train.targets.numpy().astype("int64"), x_test, test.targets.numpy().astype("int64")
    if name == "CIFAR10":
        train = datasets.CIFAR10(root=root, train=True, download=False)
        test = datasets.CIFAR10(root=root, train=False, download=False)
        x_train = np.asarray(train.data).astype("float32").transpose(0, 3, 1, 2) / 255.0
        x_test = np.asarray(test.data).astype("float32").transpose(0, 3, 1, 2) / 255.0
        return x_train, np.asarray(train.targets, dtype="int64"), x_test, np.asarray(test.targets, dtype="int64")
    raise ValueError(name)


def stratified_indices(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    for cls in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        out.extend(idx[:per_class].tolist())
    out = np.asarray(out, dtype=np.int64)
    rng.shuffle(out)
    return out


def split_pretrain_and_support_pool(
    y: np.ndarray,
    pretrain_per_class: int,
    support_pool_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pretrain = []
    pool = []
    for cls in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        pretrain.extend(idx[:pretrain_per_class].tolist())
        pool.extend(idx[pretrain_per_class : pretrain_per_class + support_pool_per_class].tolist())
    pretrain = np.asarray(pretrain, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    rng.shuffle(pretrain)
    rng.shuffle(pool)
    return pretrain, pool


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.proj(h)
        return nn.functional.normalize(z, dim=1)


class EncoderClassifier(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels, embedding_dim)
        self.head = nn.Linear(embedding_dim, n_classes)
        self.scale = 16.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        weight = nn.functional.normalize(self.head.weight, dim=1)
        return self.scale * (z @ weight.T) + self.head.bias


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_encoder(
    x: np.ndarray,
    y: np.ndarray,
    in_channels: int,
    embedding_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[ConvEncoder, float]:
    model = EncoderClassifier(in_channels, embedding_dim, 10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(x, y, batch_size=batch_size, shuffle=True)
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return model.encoder.eval(), elapsed


@torch.no_grad()
def embed(encoder: ConvEncoder, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[torch.Tensor, float]:
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    start = time.perf_counter()
    encoder.eval()
    for xb in loader:
        chunks.append(encoder(xb.to(device)).cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(chunks, dim=0), time.perf_counter() - start


def dgm_knn_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    k: int,
    temperature: float,
    batch_size: int = 2048,
) -> torch.Tensor:
    preds = []
    support_z = nn.functional.normalize(support_z, dim=1)
    query_z = nn.functional.normalize(query_z, dim=1)
    for start in range(0, len(query_z), batch_size):
        q = query_z[start : start + batch_size]
        sim = q @ support_z.T
        kk = min(k, support_z.shape[0])
        vals, idx = torch.topk(sim, k=kk, dim=1)
        weights = torch.softmax(vals / temperature, dim=1)
        logits = torch.zeros(q.shape[0], n_classes)
        labels = support_y[idx]
        logits.scatter_add_(1, labels, weights)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


def dgm_accuracy(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    k: int,
    temperature: float,
) -> tuple[float, float]:
    start = time.perf_counter()
    pred = dgm_knn_predict(support_z, support_y, test_z, 10, k=k, temperature=temperature)
    elapsed = time.perf_counter() - start
    return float((pred == test_y).float().mean().item()), elapsed


def dgm_ridge_accuracy(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    ridge: float = 1e-2,
) -> tuple[float, float]:
    start = time.perf_counter()
    support_z = nn.functional.normalize(support_z, dim=1)
    test_z = nn.functional.normalize(test_z, dim=1)
    n = support_z.shape[0]
    y_onehot = nn.functional.one_hot(support_y, num_classes=10).float()
    kernel = support_z @ support_z.T
    kernel = kernel + ridge * torch.eye(n)
    alpha = torch.linalg.solve(kernel, y_onehot)
    logits = (test_z @ support_z.T) @ alpha
    pred = torch.argmax(logits, dim=1)
    elapsed = time.perf_counter() - start
    return float((pred == test_y).float().mean().item()), elapsed


def train_linear_head(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    head = nn.Linear(support_z.shape[1], 10)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(support_z), support_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = torch.argmax(head(test_z), dim=1)
        acc = float((pred == test_y).float().mean().item())
    return acc, time.perf_counter() - start


@dataclass
class EpisodeResult:
    shots: int
    dgm_acc: float
    dgm_adapter_seconds: float
    dgm_ridge_acc: float
    dgm_ridge_seconds: float
    online_head_acc: float
    online_head_seconds: float
    tuned_head_acc: float
    tuned_head_seconds: float


def run_episode(
    support_pool_z: torch.Tensor,
    support_pool_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    shots: int,
    rng: np.random.Generator,
    seed: int,
    knn_k: int,
) -> EpisodeResult:
    support_local = []
    for cls in range(10):
        idx = torch.nonzero(support_pool_y == cls, as_tuple=False).flatten().numpy()
        rng.shuffle(idx)
        support_local.extend(idx[:shots].tolist())
    rng.shuffle(support_local)
    support_idx = torch.tensor(support_local, dtype=torch.long)
    support_z = support_pool_z[support_idx]
    support_y = support_pool_y[support_idx]

    build_start = time.perf_counter()
    support_z = support_z.contiguous()
    support_y = support_y.contiguous()
    build_seconds = time.perf_counter() - build_start
    dgm_acc, dgm_query_seconds = dgm_accuracy(
        support_z,
        support_y,
        test_z,
        test_y,
        k=min(knn_k, shots * 10),
        temperature=0.07,
    )
    dgm_ridge_acc, dgm_ridge_seconds = dgm_ridge_accuracy(
        support_z,
        support_y,
        test_z,
        test_y,
        ridge=1e-2,
    )
    online_acc, online_seconds = train_linear_head(
        support_z,
        support_y,
        test_z,
        test_y,
        epochs=1,
        lr=0.5,
        seed=seed,
    )
    tuned_acc, tuned_seconds = train_linear_head(
        support_z,
        support_y,
        test_z,
        test_y,
        epochs=200,
        lr=0.2,
        seed=seed + 1,
    )
    return EpisodeResult(
        shots=shots,
        dgm_acc=dgm_acc,
        dgm_adapter_seconds=build_seconds + dgm_query_seconds,
        dgm_ridge_acc=dgm_ridge_acc,
        dgm_ridge_seconds=build_seconds + dgm_ridge_seconds,
        online_head_acc=online_acc,
        online_head_seconds=online_seconds,
        tuned_head_acc=tuned_acc,
        tuned_head_seconds=tuned_seconds,
    )


def summarize_episode_results(rows: list[EpisodeResult]) -> dict[str, float]:
    out: dict[str, float] = {"shots": float(rows[0].shots)}
    for key in [
        "dgm_acc",
        "dgm_adapter_seconds",
        "dgm_ridge_acc",
        "dgm_ridge_seconds",
        "online_head_acc",
        "online_head_seconds",
        "tuned_head_acc",
        "tuned_head_seconds",
    ]:
        vals = np.asarray([getattr(r, key) for r in rows], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def run_dataset(
    name: str,
    root: str,
    device: torch.device,
    seed: int,
    quick: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_train, y_train, x_test, y_test = load_arrays(name, root)
    in_channels = int(x_train.shape[1])
    if quick:
        pretrain_per_class = 200 if name == "MNIST" else 400
        support_pool_per_class = 100
        test_per_class = 200
        epochs = 2 if name == "MNIST" else 5
        episodes = 2
    else:
        pretrain_per_class = 2000 if name == "MNIST" else 4000
        support_pool_per_class = 500
        test_per_class = 500
        epochs = 8 if name == "MNIST" else 30
        episodes = 5
    pre_idx, pool_idx = split_pretrain_and_support_pool(y_train, pretrain_per_class, support_pool_per_class, rng)
    test_idx = stratified_indices(y_test, test_per_class, rng)
    encoder, encoder_train_seconds = train_encoder(
        x_train[pre_idx],
        y_train[pre_idx],
        in_channels=in_channels,
        embedding_dim=128 if name == "MNIST" else 256,
        device=device,
        epochs=epochs,
        batch_size=256,
        lr=3e-3,
    )
    pool_z, pool_embed_seconds = embed(encoder, x_train[pool_idx], device, batch_size=512)
    test_z, test_embed_seconds = embed(encoder, x_test[test_idx], device, batch_size=512)
    pool_y = torch.tensor(y_train[pool_idx], dtype=torch.long)
    test_y = torch.tensor(y_test[test_idx], dtype=torch.long)
    shots_grid = [1, 5, 10] if not quick else [1, 5]
    results = {}
    for shots in shots_grid:
        rows = [
            run_episode(
                pool_z,
                pool_y,
                test_z,
                test_y,
                shots=shots,
                rng=np.random.default_rng(seed + 1000 * shots + rep),
                seed=seed + rep,
                knn_k=5,
            )
            for rep in range(episodes)
        ]
        results[f"{shots}_shot"] = summarize_episode_results(rows)
    return {
        "dataset": name,
        "device": str(device),
        "quick": quick,
        "pretrain_per_class": pretrain_per_class,
        "support_pool_per_class": support_pool_per_class,
        "test_per_class": test_per_class,
        "encoder_epochs": epochs,
        "encoder_train_seconds": encoder_train_seconds,
        "pool_embed_seconds": pool_embed_seconds,
        "test_embed_seconds": test_embed_seconds,
        "episodes": episodes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/results_dgm_adaptation.json"))
    parser.add_argument("--mnist-root", type=str, default="/Users/gunale/works/silifen-works/haoran_idea/data")
    parser.add_argument("--cifar-root", type=str, default="/Users/gunale/works/silifen-works/SLN/data/cifar10")
    args = parser.parse_args()
    set_seed(args.seed)
    device = select_device()
    results = {
        "protocol": "Shared frozen encoder. Compare DGM few-shot memory with one-pass and multi-epoch backprop heads on support embeddings.",
        "device": str(device),
        "MNIST": run_dataset("MNIST", args.mnist_root, device, args.seed + 10, args.quick),
        "CIFAR10": run_dataset("CIFAR10", args.cifar_root, device, args.seed + 20, args.quick),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"device={device}")
    for dataset in ["MNIST", "CIFAR10"]:
        table = results[dataset]
        print(f"{dataset}: encoder_train_seconds={table['encoder_train_seconds']:.2f}")
        for shot_name, row in table["results"].items():
            print(
                f"  {shot_name}: "
                f"DGM-kNN acc={row['dgm_acc_mean']:.3f}±{row['dgm_acc_std']:.3f}, "
                f"DGM-kNN={row['dgm_adapter_seconds_mean']:.4f}s; "
                f"DGM-ridge acc={row['dgm_ridge_acc_mean']:.3f}±{row['dgm_ridge_acc_std']:.3f}, "
                f"DGM-ridge={row['dgm_ridge_seconds_mean']:.4f}s; "
                f"online-head acc={row['online_head_acc_mean']:.3f}±{row['online_head_acc_std']:.3f}, "
                f"online-head={row['online_head_seconds_mean']:.4f}s; "
                f"tuned-head acc={row['tuned_head_acc_mean']:.3f}±{row['tuned_head_acc_std']:.3f}, "
                f"tuned-head={row['tuned_head_seconds_mean']:.4f}s"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
