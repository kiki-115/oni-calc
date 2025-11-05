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
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    # TorchScript
    try:
        m = torch.jit.load(str(p), map_location=device)
        m.eval()
        return m
    except Exception:
        pass

    # nn.Module
    obj = torch.load(str(p), map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # state_dict
    if isinstance(obj, dict):
        if input_size is None:
            raise RuntimeError(
                "この .pt は state_dict です。--size に学習時の入力サイズ（例: 28）を指定してください。"
            )
        model = MyCNNModel(input_size=input_size)
        model.load_state_dict(obj, strict=True)
        model.eval()
        return model

    raise RuntimeError(
        "モデルのロードに失敗しました。TorchScript/nn.Module/state_dict のいずれにも該当しません。"
    )

def preprocess_for_digit(img_path: str, size: int, channels: int, keep_ratio=True):
    img = Image.open(img_path).convert("L")

    # 反転
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    # 二値化
    arr = np.array(img)
    arr = cv2.GaussianBlur(arr, (3,3), 0)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # クロップ
    ys, xs = np.where(arr > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        arr = arr[y1:y2+1, x1:x2+1]
    
    # パディング
    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = arr

    # リサイズ
    pil = Image.fromarray(canvas)
    pil = pil.resize((size, size), Image.BICUBIC)

    # 正規化
    x = np.array(pil).astype(np.float32) / 255.0
    x = x[None, ...]
    if channels == 1:
        pass
    elif channels == 3:
        x = np.repeat(x, 3, axis=0)
    else:
        raise ValueError("channels は 1 または 3 を指定してください")
    x = torch.from_numpy(x).unsqueeze(0)
    print(torch.max(x), torch.min(x))
    return x

def predict_digit(model, img_path: str, device: torch.device, size: int, channels: int, topk: int=3):
    x = preprocess_for_digit(img_path, size=size, channels=channels).to(device)
    with torch.inference_mode():
        logits = model(x)
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
