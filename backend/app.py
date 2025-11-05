from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import io
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.inference import load_model
from utils.preprocessing import preprocess_png_like_mnist

app = FastAPI(
    title="Oni-Calc API",
    description="手書き数字認識API for 記憶力脳トレゲーム",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None

def run_model_logits(x_1x1x28x28: torch.Tensor) -> torch.Tensor:
    # CNN/MLP両対応
    try:
        return model(x_1x1x28x28)
    except Exception:
        pass
    n = x_1x1x28x28.size(0)
    x_flat = x_1x1x28x28.view(n, -1)
    return model(x_flat)

def topk_from_logits(logits: torch.Tensor, k: int = 3):
    probs = F.softmax(logits, dim=1)
    conf, idx = probs.topk(k, dim=1)
    idx = idx[0].tolist()
    conf = conf[0].tolist()
    pred = idx[0]
    p0 = conf[0]
    return idx, conf, pred, p0

@app.on_event("startup")
async def startup_event():
    # モデルロード
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/model.pt"

    try:
        model = load_model(model_path, device, input_size=28)
        model.to(device).eval()
        print(f"モデルをロードしました: {model_path}")
        print(f"デバイス: {device}")
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        raise e

@app.post("/recognize")
async def recognize_digit(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="モデルがロードされていません")

    try:
        contents = await file.read()
        size_bytes = len(contents)
        if size_bytes > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="ファイルサイズが大きすぎます（10MB制限）")

        # 画像検証
        try:
            img_probe = Image.open(io.BytesIO(contents))
            mode_hint = img_probe.mode
            w_hint, h_hint = img_probe.size
        except Exception:
            raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")

        # 前処理
        x = preprocess_png_like_mnist(contents).to(device)

        # 推論
        with torch.no_grad():
            logits = run_model_logits(x)

        idx_topk, conf_topk, pred, conf = topk_from_logits(logits, k=3)
        result_top3 = [{"digit": int(i), "prob": float(p)} for i, p in zip(idx_topk, conf_topk)]

        return JSONResponse(content={
            "success": True,
            "digit": int(pred),
            "confidence": float(conf),
            "top3": result_top3,
            "message": f"認識結果: {int(pred)} (信頼度: {conf:.3f})",
                "debug": {
                "bytes": size_bytes,
                "image_size": [int(w_hint), int(h_hint)],
                "image_mode": str(mode_hint)
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"認識エラー: {e}")
        raise HTTPException(status_code=500, detail=f"数字認識に失敗しました: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
