from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
from models.inference import load_model, preprocess_for_digit, predict_digit
import os

app = FastAPI(
    title="Oni-Calc API",
    description="手書き数字認識API for 記憶力脳トレゲーム",
    version="1.0.0"
)

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数でモデルを保持
model = None
device = None

@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルをロード"""
    global model, device
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルパス
    model_path = "models/model.pt"
    
    try:
        # モデルをロード
        model = load_model(model_path, device, input_size=28)
        model.to(device)
        print(f"モデルをロードしました: {model_path}")
        print(f"デバイス: {device}")
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        raise e

@app.get("/")
async def root():
    """ヘルスチェック用エンドポイント"""
    return {"message": "Oni-Calc API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/recognize")
async def recognize_digit(file: UploadFile = File(...)):
    """
    手書き数字画像を認識するAPI
    
    Args:
        file: アップロードされた画像ファイル
        
    Returns:
        JSON: 認識結果（数字、信頼度、上位3候補）
    """
    if model is None:
        raise HTTPException(status_code=500, detail="モデルがロードされていません")
    
    try:
        # ファイルサイズチェック（10MB制限）
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="ファイルサイズが大きすぎます（10MB制限）")
        
        # 画像ファイルかチェック
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
        
        # ファイルを読み込み
        contents = await file.read()
        
        # PIL Imageに変換
        image = Image.open(io.BytesIO(contents))
        
        # 一時ファイルとして保存（preprocess_for_digitがファイルパスを期待）
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)
        
        try:
            # 数字認識実行
            result = predict_digit(
                model=model,
                img_path=temp_path,
                device=device,
                size=28,
                channels=1,
                topk=3
            )
            
            return JSONResponse(content={
                "success": True,
                "digit": result["pred"],
                "confidence": result["conf"],
                "top3": result["topk"],
                "message": f"認識結果: {result['pred']} (信頼度: {result['conf']:.3f})"
            })
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"認識エラー: {e}")
        raise HTTPException(status_code=500, detail=f"数字認識に失敗しました: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
