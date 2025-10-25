import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentProblem, setCurrentProblem] = useState(null);
  const [memoryAnswers, setMemoryAnswers] = useState([]);
  const [score, setScore] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const [n, setN] = useState(3); // n個前の問題の答えを記憶

  // 問題生成
  const generateProblem = () => {
    const operations = ['+', '-'];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    
    let num1, num2, answer;
    if (operation === '+') {
      num1 = Math.floor(Math.random() * 9) + 1;
      num2 = Math.floor(Math.random() * (9 - num1)) + 1;
      answer = num1 + num2;
    } else {
      num1 = Math.floor(Math.random() * 9) + 1;
      num2 = Math.floor(Math.random() * num1) + 1;
      answer = num1 - num2;
    }
    
    return { num1, num2, operation, answer };
  };

  // ゲーム開始
  const startGame = () => {
    setGameStarted(true);
    setCurrentProblem(generateProblem());
    setMemoryAnswers([]);
    setScore(0);
  };

  // Canvas描画開始
  const startDrawing = (e) => {
    if (!canvasRef.current) return;
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 描画設定
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';
    
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  };

  // Canvas描画中
  const draw = (e) => {
    if (!isDrawing || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 描画設定（毎回設定）
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';
    
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  };

  // Canvas描画終了
  const stopDrawing = () => {
    setIsDrawing(false);
  };

  // Canvasクリア
  const clearCanvas = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 背景を白に設定
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 描画設定を再設定
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';
    
    console.log('Canvas cleared');
  };

  // 数字認識API呼び出し
  const recognizeDigit = async () => {
    if (!canvasRef.current) return null;
    const canvas = canvasRef.current;
    
    // Canvasの内容を確認
    console.log('Canvas size:', canvas.width, 'x', canvas.height);
    
    // Canvasの内容を画像として確認
    const dataURL = canvas.toDataURL('image/png');
    console.log('DataURL length:', dataURL.length);
    console.log('DataURL preview:', dataURL.substring(0, 100) + '...');
    
    // Base64をBlobに変換
    const response = await fetch(dataURL);
    const blob = await response.blob();
    console.log('Blob size:', blob.size, 'bytes');
    
    // FormDataを作成
    const formData = new FormData();
    formData.append('file', blob, 'digit.png');
    
    try {
      console.log('Sending request to API...');
      const apiResponse = await fetch('http://localhost:8000/recognize', {
        method: 'POST',
        body: formData,
      });
      
      const result = await apiResponse.json();
      console.log('API response:', result);
      return result.digit;
    } catch (error) {
      console.error('数字認識エラー:', error);
      return null;
    }
  };

  // 答えを送信
  const submitAnswer = async () => {
    const recognizedDigit = await recognizeDigit();
    if (recognizedDigit === null) return;
    
    console.log('記憶中の答え:', memoryAnswers);
    console.log('現在のn:', n);
    console.log('認識した数字:', recognizedDigit);
    
    // n個前の問題の答えと比較
    if (memoryAnswers.length >= n) {
      const targetAnswer = memoryAnswers[memoryAnswers.length - n];
      console.log('比較対象の答え:', targetAnswer);
      
      if (recognizedDigit === targetAnswer) {
        setScore(score + 1);
        alert(`正解！+1点 (現在のスコア: ${score + 1})`);
      } else {
        alert(`不正解... 正解は ${targetAnswer} でした (現在のスコア: ${score})`);
      }
    } else {
      alert(`まだ${n}個の問題がありません。現在: ${memoryAnswers.length}個`);
    }
    
    // 新しい問題を生成
    const newProblem = generateProblem();
    setCurrentProblem(newProblem);
    
    // 記憶リストに追加
    setMemoryAnswers([...memoryAnswers, newProblem.answer]);
    
    // Canvasをクリア
    clearCanvas();
  };

  // Canvas初期化
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 背景を白に設定
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 描画設定
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';
    
    console.log('Canvas initialized');
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Oni-Calc: 記憶力脳トレゲーム</h1>
        
        {!gameStarted ? (
          <div>
            <p>n個前の問題の答えを記憶して手書きで入力するゲームです</p>
            <label>
              nの値: 
              <input 
                type="number" 
                value={n} 
                onChange={(e) => setN(parseInt(e.target.value))}
                min="1" 
                max="10"
              />
            </label>
            <br />
            <button onClick={startGame}>ゲーム開始</button>
          </div>
        ) : (
          <div>
            <div className="game-info">
              <p>スコア: {score}</p>
              <p>記憶中の答え: {memoryAnswers.slice(-n).join(', ')}</p>
            </div>
            
            <div className="problem">
              <h2>問題: {currentProblem?.num1} {currentProblem?.operation} {currentProblem?.num2} = ?</h2>
              <p>※{n}個前の問題の答えを手書きで入力してください</p>
            </div>
            
            <div className="canvas-container">
              <canvas
                ref={canvasRef}
                width={300}
                height={300}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                style={{ border: '1px solid #000', backgroundColor: 'white' }}
              />
            </div>
            
            <div className="controls">
              <button onClick={clearCanvas}>クリア</button>
              <button onClick={() => {
                // テスト用：数字「7」を描画
                if (!canvasRef.current) return;
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(100, 50);
                ctx.lineTo(100, 200);
                ctx.moveTo(100, 50);
                ctx.lineTo(150, 50);
                ctx.moveTo(100, 125);
                ctx.lineTo(150, 125);
                ctx.stroke();
                console.log('Test digit 7 drawn');
              }}>テスト描画(7)</button>
              <button onClick={submitAnswer}>送信</button>
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;