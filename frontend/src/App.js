import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import {
  Layout, Row, Col, Card, Space, Typography,
  Button, Statistic, Tag, InputNumber, Divider,
  App as AntdApp, Modal,
} from 'antd';
import { SendOutlined, DeleteOutlined, UpOutlined, DownOutlined, HomeOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

/** 0–9 に収まる一桁演算のみ生成（＋は a+b<=9、−は a>=b） */
function generateProblem() {
  const op = Math.random() < 0.5 ? '+' : '-';
  while (true) {
    const a = Math.floor(Math.random() * 10); // 0..9
    const b = Math.floor(Math.random() * 10);
    if (op === '+') {
      if (a + b <= 9) return { num1: a, num2: b, operation: '+', answer: a + b };
    } else {
      if (a - b >= 0) return { num1: a, num2: b, operation: '-', answer: a - b };
    }
  }
}

export default function App() {
  // AntD v5: messageはAppコンテキストから取得
  const { message } = AntdApp.useApp();

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentProblem, setCurrentProblem] = useState(null);
  const [memoryAnswers, setMemoryAnswers] = useState([]); // 末尾が最新
  const [score, setScore] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const [n, setN] = useState(3);
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  
  // 問題履歴を管理（表示用）
  const [problemHistory, setProblemHistory] = useState([]); // 末尾が最新
  const [questionNumber, setQuestionNumber] = useState(0); // ゲーム全体の通し番号
  const [showJudgment, setShowJudgment] = useState(false); // 判定を表示するか
  const [pendingJudgment, setPendingJudgment] = useState(null); // 判定中の問題
  const [hintModalVisible, setHintModalVisible] = useState(false); // ヒントモーダルの表示状態

  /** ゲーム開始 */
  const startGame = () => {
    const p = generateProblem();
    setGameStarted(true);
    setCurrentProblem(p);
    setMemoryAnswers([]);        // 最初は覚えるだけ
    setProblemHistory([p]);       // 問題履歴を初期化（最初の問題のみ）
    setQuestionNumber(1);         // 1問目から開始
    setShowJudgment(false);
    setPendingJudgment(null);
    setScore(0);
    
    // Canvas初期化
    setTimeout(() => {
      if (!canvasRef.current) {
        console.error('Canvas ref is null');
        return;
      }
      
      const canvas = canvasRef.current;
      const cssW = 400, cssH = 400;
      canvas.width = Math.floor(cssW * dpr);   // 内部ピクセル
      canvas.height = Math.floor(cssH * dpr);
      canvas.style.width = cssW + 'px';
      canvas.style.height = cssH + 'px';
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      console.log('Canvas initialized:', {
        width: canvas.width,
        height: canvas.height,
        aspectRatio: canvas.width / canvas.height
      });
    }, 100);
  };

  /** ゲームをリセットして最初に戻す */
  const restartGame = () => {
    const p = generateProblem();
    setCurrentProblem(p);
    setMemoryAnswers([]);        // 最初は覚えるだけ
    setProblemHistory([p]);       // 問題履歴を初期化（最初の問題のみ）
    setQuestionNumber(1);         // 1問目から開始
    setShowJudgment(false);
    setPendingJudgment(null);
    setScore(0);
    
    // Canvas初期化
    setTimeout(() => {
      if (!canvasRef.current) {
        console.error('Canvas ref is null');
        return;
      }
      
      const canvas = canvasRef.current;
      const cssW = 400, cssH = 400;
      canvas.width = Math.floor(cssW * dpr);   // 内部ピクセル
      canvas.height = Math.floor(cssH * dpr);
      canvas.style.width = cssW + 'px';
      canvas.style.height = cssH + 'px';
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      console.log('Canvas initialized:', {
        width: canvas.width,
        height: canvas.height,
        aspectRatio: canvas.width / canvas.height
      });
    }, 100);
    
    message.info('ゲームをリセットしました');
  };

  /** CSS座標 → 内部ピクセル座標に変換 */
  const getCanvasPoint = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  /** Canvas描画（Pointer Events） */
  const startDrawing = (e) => {
    e.preventDefault();
    if (!canvasRef.current || memoryAnswers.length < n) return;
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.lineWidth = 10 * dpr;     // DPI対応
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';

    const { x, y } = getCanvasPoint(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    e.preventDefault();
    if (!isDrawing || !canvasRef.current || memoryAnswers.length < n) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.lineWidth = 10 * dpr;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';

    const { x, y } = getCanvasPoint(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const stopDrawing = () => setIsDrawing(false);

  const clearCanvas = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    // 内部ピクセル単位で白塗り
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  /** 数字認識 API */
  const recognizeDigit = async () => {
    if (!canvasRef.current) return null;
    const canvas = canvasRef.current;

    // デバッグ: 実サイズ(属性)と見た目(CSS)を確認
    const rect = canvas.getBoundingClientRect();
    console.log('[canvas]', {
      dpr,
      attr: { w: canvas.width, h: canvas.height },
      css: { w: rect.width, h: rect.height },
    });

    const dataURL = canvas.toDataURL('image/png');
    const res = await fetch(dataURL);
    const blob = await res.blob();
    const formData = new FormData();
    formData.append('file', blob, 'digit.png');

    try {
      const apiResponse = await fetch('http://localhost:8000/recognize', {
        method: 'POST',
        body: formData,
      });
      const result = await apiResponse.json();
      return typeof result.digit === 'number' ? result.digit : null;
    } catch (e) {
      console.error(e);
      message.error('認識APIに接続できませんでした');
      return null;
    }
  };

  /** 送信 */
  const submitAnswer = async () => {
    if (!currentProblem || memoryAnswers.length < n) return;

    const recognizedDigit = await recognizeDigit();
    if (recognizedDigit === null) return;

    // n個前の問題を取得（履歴の最初の問題）
    const targetProblem = problemHistory[0];
    const targetAnswer = memoryAnswers[memoryAnswers.length - n];
    
    // 推論結果を問題履歴に反映
    const newHistory = [...problemHistory];
    newHistory[0] = {
      ...problemHistory[0],
      recognizedAnswer: recognizedDigit,
      isCorrect: recognizedDigit === targetAnswer
    };
    setProblemHistory(newHistory);
    
    // 推論結果を即座に反映（まだ判定は表示しない）
    setPendingJudgment({ 
      recognizedAnswer: recognizedDigit, 
      isCorrect: recognizedDigit === targetAnswer 
    });
    
    // 判定を0.2秒遅延して表示
    setTimeout(() => {
      setShowJudgment(true);
      
      if (recognizedDigit === targetAnswer) {
        setScore((s) => s + 1);
        message.success('正解！ +1点');
      } else {
        message.error(`不正解… 正解は ${targetAnswer}`);
      }
      
      // さらに0.3秒後に次の問題へ
      setTimeout(() => {
        setPendingJudgment(null);
        setShowJudgment(false);
        
        const newProblem = generateProblem();
        setMemoryAnswers((prev) => [...prev, currentProblem.answer]);
        setCurrentProblem(newProblem);
        setQuestionNumber((prev) => prev + 1); // 問題番号をインクリメント
        // 履歴の先頭を削除し、最後尾に追加（n個固定）
        setProblemHistory((prev) => {
          const updated = [...prev];
          updated.shift(); // 最古の問題を削除
          updated.push(newProblem); // 新しい問題を追加
          return updated;
        });
        clearCanvas();
      }, 300);
    }, 200);
  };

  // 記憶中は1.5秒ごとに自動で問題が切り替わる
  useEffect(() => {
    if (!gameStarted || memoryAnswers.length >= n) return;

    const timer = setInterval(() => {
      setMemoryAnswers((prev) => [...prev, currentProblem.answer]);
      const newProblem = generateProblem();
      setCurrentProblem(newProblem);
      setProblemHistory((prev) => [...prev, newProblem]);
      setQuestionNumber((prev) => prev + 1);
      clearCanvas();
    }, 1500);

    return () => clearInterval(timer);
  }, [gameStarted, memoryAnswers.length, n, currentProblem?.answer]);

  return (
    <Layout className="app-wrap">
      <Header style={{ background: '#0b1220' }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={3} style={{ color: '#e6f4ff', margin: 0 }}>
              Oni-Calc: 記憶力トレーニング
            </Title>
          </Col>
        </Row>
      </Header>

      <Content style={{ padding: 24 }}>
        {!gameStarted ? (
          <Card className="board">
            <Text style={{ color: '#cbd5e1' }}>
              {n}個前の問題の答えを、手書きで入力するゲームです。最初の {n} 問は覚えるだけ。
            </Text>
            <Divider />
            <Space size="large" align="center">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 0, alignItems: 'center' }}>
                  <Button 
                    type="text" 
                    icon={<UpOutlined />} 
                    onClick={() => setN(prev => Math.min(9, prev + 1))}
                    style={{ height: '20px', padding: 0 }}
                  />
                  <Text style={{ 
                    color: '#fff', 
                    fontSize: 20, 
                    fontWeight: 'bold',
                    lineHeight: '24px'
                  }}>
                    {n}
                  </Text>
                  <Button 
                    type="text" 
                    icon={<DownOutlined />} 
                    onClick={() => setN(prev => Math.max(1, prev - 1))}
                    style={{ height: '20px', padding: 0 }}
                  />
                </div>
                <Text style={{ color: '#cbd5e1', fontSize: 16 }}>バック</Text>
              </div>
              <Button type="primary" onClick={startGame}>ゲーム開始</Button>
            </Space>
          </Card>
        ) : (
          <Row gutter={[24, 24]}>
            <Col xs={24} md={12}>
              <Card className="board" styles={{ body: { padding: 20 } }}>
                <Row gutter={[16, 16]}>
                  <Col xs={12} md={6}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Button 
                        type="text"
                        icon={<QuestionCircleOutlined />}
                        onClick={() => setHintModalVisible(true)}
                        style={{ color: '#94a3b8' }}
                      />
                      <Statistic title="スコア" value={score} valueStyle={{ color: '#f0f9ff' }} />
                    </div>
                  </Col>
                  <Col xs={12} md={18}>
                    <div style={{ position: 'relative', minHeight: '32px' }}>
                      <Text 
                        className={memoryAnswers.length < n ? 'telegraph-blink' : ''}
                        style={{ 
                          color: memoryAnswers.length < n 
                            ? '#fbbf24' 
                            : memoryAnswers.length === n 
                              ? '#ef4444' 
                              : '#cbd5e1', 
                          fontSize: 20,
                          fontWeight: 500,
                          position: 'absolute',
                          left: 0,
                          bottom: 0
                        }}
                      >
                        {memoryAnswers.length < n 
                          ? 'いきますよ…鬼計算…' 
                          : memoryAnswers.length === n 
                            ? 'はじめ！！' 
                            : ''}
                      </Text>
                      <Button 
                        icon={<HomeOutlined />} 
                        onClick={restartGame}
                        style={{ 
                          position: 'absolute', 
                          right: 120, 
                          top: 0,
                          transform: 'translateY(-8px)'
                        }}
                        size="small"
                      >
                        やり直す
                      </Button>
                      <Button 
                        icon={<HomeOutlined />} 
                        onClick={() => setGameStarted(false)}
                        style={{ 
                          position: 'absolute', 
                          right: 0, 
                          top: 0,
                          transform: 'translateY(-8px)'
                        }}
                        size="small"
                      >
                        ホーム
                      </Button>
                    </div>
                  </Col>
                </Row>

                <Divider style={{ borderColor: '#1f2937' }} />

                {/* 2つの式を表示：現在の問題 + n個前の問題 */}
                <div style={{ marginBottom: 24 }}>
                  {/* 上方向の点線 */}
                  {problemHistory.length > n && (
                    <div style={{ 
                      textAlign: 'center', 
                      padding: '4px 0', 
                      color: '#9ca3af',
                      fontSize: '24px',
                      fontWeight: 'bold',
                      letterSpacing: '3px'
                    }}>
                      ⋮
                    </div>
                  )}

                  {/* 現在の問題（表示） */}
                  <div style={{ marginBottom: 16, padding: 16, background: '#1f2937', borderRadius: 8 }}>
                    <Text style={{ color: '#9ca3af' }}>
                      {questionNumber}問目
                    </Text>
                    <Title 
                      key={`current-${questionNumber}`}
                      level={3} 
                      className={!pendingJudgment ? "slide-in-text" : ""} 
                      style={{ color: '#fff', margin: 0 }}
                    >
                      {currentProblem?.num1} {currentProblem?.operation} {currentProblem?.num2} = ?
                    </Title>
                  </div>

                  {/* 中間の点線 */}
                  {problemHistory.length > n && (
                    <div style={{ 
                      textAlign: 'center', 
                      padding: '4px 0', 
                      color: '#9ca3af',
                      fontSize: '24px',
                      fontWeight: 'bold',
                      letterSpacing: '3px'
                    }}>
                      ⋮
                    </div>
                  )}

                  {/* n個前の問題（問題履歴の最初）- i番目をマスク */}
                  {problemHistory.length > n ? (
                    <div style={{ 
                      marginBottom: 16, 
                      padding: 16, 
                      background: pendingJudgment && showJudgment ? 
                        (pendingJudgment.isCorrect ? '#14532d' : '#7f1d1d') : 
                        '#1f2937',
                      borderRadius: 8,
                      border: pendingJudgment && showJudgment ? 
                        (pendingJudgment.isCorrect ? '2px solid #22c55e' : '2px solid #ef4444') : 
                        '1px solid #374151',
                      transition: 'all 0.3s'
                    }}>
                      <Text style={{ color: '#9ca3af' }}>
                        {questionNumber - n}問目
                      </Text>
                      <Title 
                        key={`history-${questionNumber - n}-${pendingJudgment ? 'answered' : 'pending'}`}
                        level={3} 
                        className={!pendingJudgment ? "slide-in-text" : ""} 
                        style={{ color: '#fff', margin: 0 }}
                      >
                        {pendingJudgment ? (
                          // 文字を埋めたらマスクを外して式を表示
                          <>
                            {problemHistory[0]?.num1} {problemHistory[0]?.operation} {problemHistory[0]?.num2} = 
                            <span style={{
                              color: pendingJudgment.isCorrect ? '#22c55e' : '#ef4444',
                              fontWeight: 'bold',
                              display: 'inline-block',
                              width: '30px',
                              height: '30px',
                              lineHeight: '30px',
                              background: 'rgba(255, 255, 255, 0.1)',
                              borderRadius: '4px',
                              textAlign: 'center'
                            }}>
                              {pendingJudgment.recognizedAnswer}
                            </span>
                          </>
                        ) : (
                          // まだ記入していない時はマスク表示
                          <>
                            ? <span style={{ fontSize: '0.7em' }}>?</span> ? = 
                            <span style={{
                              display: 'inline-block',
                              width: '30px',
                              height: '30px',
                              background: 'rgba(255, 255, 255, 0.1)',
                              borderRadius: '4px'
                            }}></span>
                          </>
                        )}
                        {pendingJudgment && showJudgment && (
                          <span style={{
                            marginLeft: 8,
                            fontSize: 24,
                            color: pendingJudgment.isCorrect ? '#22c55e' : '#ef4444',
                            fontWeight: 'bold'
                          }}>
                            {pendingJudgment.isCorrect ? '○' : '✗'}
                          </span>
                        )}
                      </Title>
                    </div>
                  ) : (
                    <Text style={{ color: '#6b7280', fontStyle: 'italic' }}>
                      最初の{n}問は覚えるだけです...
                    </Text>
                  )}

                  {/* 下方向の点線 */}
                  {problemHistory.length > n && (
                    <div style={{ 
                      textAlign: 'center', 
                      padding: '4px 0', 
                      color: '#9ca3af',
                      fontSize: '24px',
                      fontWeight: 'bold',
                      letterSpacing: '3px'
                    }}>
                      ⋮
                    </div>
                  )}
                </div>

              </Card>
            </Col>

            <Col xs={24} md={12}>
              <Card className="board" styles={{ body: { padding: 20 } }}>
                <Space direction="vertical" style={{ width: '100%' }} size="large">
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <div style={{ 
                      opacity: memoryAnswers.length < n ? 0.4 : 1, 
                      pointerEvents: memoryAnswers.length < n ? 'none' : 'auto' 
                    }}>
                      <canvas
                        ref={canvasRef}
                        className="pad"
                        onPointerDown={startDrawing}
                        onPointerMove={draw}
                        onPointerUp={stopDrawing}
                        onPointerLeave={stopDrawing}
                        onContextMenu={(e) => e.preventDefault()}
                        style={{ touchAction: 'none' }}  // タッチデバイスのスクロール抑止
                        // width/height は useEffect で設定（HiDPI）
                      />
                    </div>
                  </div>
                  <Space wrap style={{ width: '100%', justifyContent: 'center' }}>
                    <Button 
                      danger 
                      icon={<DeleteOutlined />} 
                      onClick={clearCanvas}
                      disabled={memoryAnswers.length < n}
                    >
                      消す
                    </Button>
                    <Button 
                      type="primary" 
                      icon={<SendOutlined />} 
                      onClick={submitAnswer}
                      disabled={memoryAnswers.length < n}
                      style={{ opacity: memoryAnswers.length < n ? 0.5 : 1 }}
                    >
                      送信
                    </Button>
                  </Space>
                </Space>
              </Card>
            </Col>
          </Row>
        )}

        {/* ヒントモーダル */}
        <Modal
          title="ヒント"
          open={hintModalVisible}
          onCancel={() => setHintModalVisible(false)}
          footer={null}
          styles={{ content: { background: '#1f2937' }, header: { background: '#1f2937', color: '#e5e7eb', borderBottom: '1px solid #374151' } }}
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text style={{ color: '#cbd5e1' }}>・最初の <b>{n}</b> 問は覚えるだけ（自動で次へ進みます）</Text>
            <Text style={{ color: '#cbd5e1' }}>・答えは 0–9 の一桁のみ</Text>
          </Space>
        </Modal>
      </Content>
    </Layout>
  );
}
