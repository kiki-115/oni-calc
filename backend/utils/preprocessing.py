"""
画像前処理モジュール
MNIST形式への変換処理を含む
"""
import io
import numpy as np
import torch
from PIL import Image, ImageOps


def preprocess_png_like_mnist(png_bytes: bytes) -> torch.Tensor:
    """
    PNGバイト列をMNIST形式のTensorに変換する前処理関数
    
    RGBA -> 白合成 -> グレースケール -> 反転(白字) -> BBox切出 ->
    長辺20pxへ等比縮小 -> 28x28パディング -> 重心センタリング -> [0,1]Tensor (1,1,28,28)
    
    Args:
        png_bytes: PNG画像のバイト列
        
    Returns:
        torch.Tensor: (1, 1, 28, 28)の形状、値は[0, 1]の範囲
    """
    # 1) 読み込み & 白背景合成 -> グレースケール
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")  # 8bit gray

    # 2) MNISTは黒地に白字想定なので反転（白地黒字→黒地白字）
    img = ImageOps.invert(img)
    arr = np.array(img, dtype=np.uint8)

    # 3) 前景のBBox（0より明るい画素を前景）
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        # 何も書いてない場合
        x = np.zeros((1, 1, 28, 28), dtype=np.float32)
        return torch.from_numpy(x)

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    crop = arr[ymin:ymax + 1, xmin:xmax + 1]

    # 4) 長辺を20pxに等比縮小（BICUBIC）
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.BICUBIC)

    # 5) 28x28の黒キャンバスに中央配置
    canvas = Image.new("L", (28, 28), 0)
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2
    canvas.paste(crop_img, (ox, oy))

    # 6) 重心センタリング（1px単位の整数シフト）
    a = np.array(canvas, dtype=np.float32)
    y_idx, x_idx = np.mgrid[0:28, 0:28]
    mass = a.sum() + 1e-6
    cx = (a * x_idx).sum() / mass
    cy = (a * y_idx).sum() / mass
    sx = int(round(14 - cx))
    sy = int(round(14 - cy))
    a = np.roll(np.roll(a, sy, axis=0), sx, axis=1)

    # 7) 0–1へ正規化 & (1,1,28,28)
    a = (a / 255.0).astype(np.float32)
    a = a[None, None, :, :]
    return torch.from_numpy(a)

