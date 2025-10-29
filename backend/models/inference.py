import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import cv2
from .model import MyCNNModel

def load_model(model_path: str, device: torch.device, input_size: int | None = None):
    """
    .pt が TorchScript（torch.jit.load）または nn.Module（torch.load）に対応。
    どちらでも試して成功した方を返す。
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    # 1) TorchScript (最優先: 依存なしでロードできる)
    try:
        m = torch.jit.load(str(p), map_location=device)
        m.eval()
        return m
    except Exception:
        pass

    # 2) nn.Module が丸ごと保存されている場合
    # PyTorch 2.6 以降は既定で weights_only=True のため、明示的に False にする
    obj = torch.load(str(p), map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # 3) state_dict の場合は MyModel で復元を試す
    if isinstance(obj, dict):
        if input_size is None:
            raise RuntimeError(
                "この .pt は state_dict です。--size に学習時の入力サイズ（例: 28）を指定してください。"
            )
        model = MyCNNModel(input_size=input_size)
        model.load_state_dict(obj, strict=True)
        model.eval()
        return model

    # それ以外
    raise RuntimeError(
        "モデルのロードに失敗しました。TorchScript/nn.Module/state_dict のいずれにも該当しません。"
    )

def preprocess_for_digit(img_path: str, size: int, channels: int, keep_ratio=True):
    """
    実運用での取りやすい前処理：
      - グレースケール
      - 背景が白・文字が黒なら自動反転（MNIST 互換: 黒背景に白文字がデフォ）
      - 28x28 等にリサイズ（余白パディングして中央寄せ）
      - 0-1 正規化 & （MNIST なら）標準化
    """
    # 読み込み（PIL）
    img = Image.open(img_path).convert("L")  # grayscale

    # 自動反転（平均が明るい＝白背景っぽい → 反転して「黒背景・白文字」へ）
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    # OpenCV で軽い二値化（ノイズ抑制 & コントラスト改善）
    arr = np.array(img)
    arr = cv2.GaussianBlur(arr, (3,3), 0)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 文字の外接矩形でクロップ（空白を削る）
    ys, xs = np.where(arr > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        arr = arr[y1:y2+1, x1:x2+1]
    # 正方形キャンバスにパディングして中央寄せ
    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = arr

    # リサイズ
    pil = Image.fromarray(canvas)
    pil = pil.resize((size, size), Image.BICUBIC)

    # Tensor 化
    x = np.array(pil).astype(np.float32) / 255.0  # [0,1]
    x = x[None, ...]  # (1,H,W)
    if channels == 1:
        pass
    elif channels == 3:
        x = np.repeat(x, 3, axis=0)  # (3,H,W)
    else:
        raise ValueError("channels は 1 または 3 を指定してください")
    x = torch.from_numpy(x).unsqueeze(0)  # (1,C,H,W)
    print(torch.max(x), torch.min(x))
    return x

def predict_digit(model, img_path: str, device: torch.device, size: int, channels: int, topk: int=3):
    x = preprocess_for_digit(img_path, size=size, channels=channels).to(device)
    with torch.inference_mode():
        logits = model(x)  # 期待: (1,10)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)
        prob, pred = probs.max(dim=-1)
        topv, topi = probs.topk(k=min(topk, probs.shape[-1]), dim=-1)
    return {
        "pred": int(pred.item()),
        "conf": float(prob.item()),
        "topk": [(int(i), float(v)) for v, i in zip(topv[0].tolist(), topi[0].tolist())]
    }

def main():
    ap = argparse.ArgumentParser(description="Handwritten digit inference (local)")
    ap.add_argument("--model", required=True, help="path to .pt")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--size", type=int, default=28, help="model input size (e.g., 28 or 224)")
    ap.add_argument("--channels", type=int, default=1, help="1 or 3 (model input channels)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(args.model, device, input_size=args.size)
    model.to(device)
    out = predict_digit(model, args.image, device, size=args.size, channels=args.channels, topk=3)
    print(f"prediction={out['pred']}  conf={out['conf']:.3f}  top3={out['topk']}")

if __name__ == "__main__":
    main()
