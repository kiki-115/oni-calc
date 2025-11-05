import io
import numpy as np
import torch
from PIL import Image, ImageOps


def preprocess_png_like_mnist(png_bytes: bytes) -> torch.Tensor:
    # 白背景合成
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    # 反転
    img = ImageOps.invert(img)
    arr = np.array(img, dtype=np.uint8)

    # クロップ
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        x = np.zeros((1, 1, 28, 28), dtype=np.float32)
        return torch.from_numpy(x)

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    crop = arr[ymin:ymax + 1, xmin:xmax + 1]

    # リサイズ
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.BICUBIC)

    # パディング
    canvas = Image.new("L", (28, 28), 0)
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2
    canvas.paste(crop_img, (ox, oy))

    # 重心センタリング
    a = np.array(canvas, dtype=np.float32)
    y_idx, x_idx = np.mgrid[0:28, 0:28]
    mass = a.sum() + 1e-6
    cx = (a * x_idx).sum() / mass
    cy = (a * y_idx).sum() / mass
    sx = int(round(14 - cx))
    sy = int(round(14 - cy))
    a = np.roll(np.roll(a, sy, axis=0), sx, axis=1)

    # 正規化
    a = (a / 255.0).astype(np.float32)
    a = a[None, None, :, :]
    return torch.from_numpy(a)

