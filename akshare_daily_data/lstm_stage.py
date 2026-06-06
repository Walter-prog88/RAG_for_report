from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .baostock_daily import fetch_a_share_daily_baostock
    from .factor_stage import build_factor_dataframe
except ImportError:
    from baostock_daily import fetch_a_share_daily_baostock
    from factor_stage import build_factor_dataframe


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "mom_20d",
    "vol_ratio_5",
    "turn_ma_20",
    "price_ma_ratio",
]
TARGET_COLUMN = "y_future_5d"


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2022-08-01"
    train_end: str = "2024-12-31"
    val_start: str = "2025-01-01"
    val_end: str = "2025-08-31"
    test_start: str = "2025-09-01"
    test_end: str = "2026-05-15"


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask].copy().reset_index(drop=True)


def standardize_by_train(
    train_df: pd.DataFrame,
    *frames: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    mean = train_df[FEATURE_COLUMNS].mean()
    std = train_df[FEATURE_COLUMNS].std(ddof=0).replace(0, np.nan)
    std = std.fillna(1.0)

    standardized = []
    for frame in frames:
        out = frame.copy()
        out[FEATURE_COLUMNS] = (out[FEATURE_COLUMNS] - mean) / std
        standardized.append(out)
    return tuple(standardized)


def make_windows(df: pd.DataFrame, window: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(df) < window:
        return (
            np.empty((0, window, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()

    xs, ys, sample_dates = [], [], []
    for end_idx in range(window - 1, len(df)):
        xs.append(features[end_idx - window + 1 : end_idx + 1])
        ys.append(target[end_idx])
        sample_dates.append(dates[end_idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), sample_dates


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    preds, ys = [], []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total_loss += float(criterion(pred, yb).item())
            total_count += len(yb)
            preds.append(pred.cpu().numpy())
            ys.append(yb.cpu().numpy())
    if total_count == 0:
        return float("nan"), np.array([]), np.array([])
    return total_loss / total_count, np.concatenate(preds), np.concatenate(ys)


def pearson_ic(pred: np.ndarray, y: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def describe_split(name: str, frame: pd.DataFrame, sample_dates: list[str]) -> dict:
    if frame.empty:
        raw_range = [None, None]
    else:
        raw_range = [
            str(pd.to_datetime(frame["date"]).min().date()),
            str(pd.to_datetime(frame["date"]).max().date()),
        ]
    if sample_dates:
        sample_range = [sample_dates[0], sample_dates[-1]]
    else:
        sample_range = [None, None]
    return {
        "name": name,
        "factor_rows": len(frame),
        "window_samples": len(sample_dates),
        "factor_date_range": raw_range,
        "sample_date_range": sample_range,
    }


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    model_path: Path,
) -> tuple[LSTMRegressor, list[dict]]:
    model = LSTMRegressor(input_size=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_count += len(yb)

        train_mse = train_loss_sum / train_count
        val_mse, _, _ = evaluate(model, val_loader, device)
        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            torch.save(model.state_dict(), model_path)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "best": improved})
        print(f"epoch={epoch:02d} train_mse={train_mse:.8f} val_mse={val_mse:.8f} best={improved}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM on BaoStock A-share factor data.")
    parser.add_argument("--symbol", default="300308", help="A-share code, e.g. 300308")
    parser.add_argument("--data-start", default="20220801", help="Raw data fetch start date.")
    parser.add_argument("--data-end", default="20260515", help="Raw data fetch end date.")
    parser.add_argument("--adjustflag", default="2", choices=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="akshare_daily_data/lstm_outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.symbol}_best_lstm.pt"
    metrics_path = output_dir / f"{args.symbol}_lstm_metrics.json"
    predictions_path = output_dir / f"{args.symbol}_test_predictions.csv"

    raw_df = fetch_a_share_daily_baostock(
        args.symbol,
        start_date=args.data_start,
        end_date=args.data_end,
        adjustflag=args.adjustflag,
    )
    factor_df = build_factor_dataframe(raw_df)

    requested_end = pd.to_datetime(args.data_end).date()
    raw_end = pd.to_datetime(raw_df["date"]).max().date() if not raw_df.empty else None
    if raw_end is not None and raw_end < requested_end:
        print(
            f"warning=BaoStock returned daily bars only through {raw_end}; "
            f"requested data_end={requested_end}. Same-day daily bars may not be loaded yet."
        )

    split_cfg = SplitConfig()
    train_df = split_frame(factor_df, split_cfg.train_start, split_cfg.train_end)
    val_df = split_frame(factor_df, split_cfg.val_start, split_cfg.val_end)
    test_df = split_frame(factor_df, split_cfg.test_start, split_cfg.test_end)

    train_std, val_std, test_std = standardize_by_train(train_df, train_df, val_df, test_df)

    x_train, y_train, train_dates = make_windows(train_std, args.window)
    x_val, y_val, val_dates = make_windows(val_std, args.window)
    x_test, y_test, test_dates = make_windows(test_std, args.window)
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise RuntimeError("Not enough rows to build train/val/test windows. Check date ranges and window size.")

    split_info = [
        describe_split("train", train_df, train_dates),
        describe_split("validation", val_df, val_dates),
        describe_split("test", test_df, test_dates),
    ]
    print(json.dumps({"split_info": split_info}, ensure_ascii=False, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model, history = train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_path=model_path,
    )
    test_mse, pred, y_true = evaluate(model, test_loader, device)
    ic = pearson_ic(pred, y_true)

    pd.DataFrame({
        "date": test_dates,
        "prediction": pred,
        "y_true": y_true,
    }).to_csv(predictions_path, index=False)

    metrics = {
        "symbol": args.symbol,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "window": args.window,
        "model": {
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": 64,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.2,
        },
        "training": {
            "loss": "MSE",
            "optimizer": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "best_model_path": str(model_path),
        },
        "splits": asdict(split_cfg),
        "split_info": split_info,
        "test": {
            "mse": float(test_mse),
            "ic": ic,
            "prediction_stats": stats(pred),
            "y_true_stats": stats(y_true),
            "predictions_path": str(predictions_path),
        },
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Test evaluation")
    print("=" * 72)
    print(f"MSE                 : {test_mse:.8f}")
    print(f"IC                  : {ic:.6f}")
    print(f"prediction stats    : {metrics['test']['prediction_stats']}")
    print(f"y_true stats        : {metrics['test']['y_true_stats']}")
    print(f"best model          : {model_path}")
    print(f"metrics             : {metrics_path}")
    print(f"test predictions    : {predictions_path}")


if __name__ == "__main__":
    main()
