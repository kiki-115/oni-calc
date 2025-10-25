import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import {
  Layout, Row, Col, Card, Space, Typography,
  Button, Statistic, Tag, InputNumber, Divider,
  App as AntdApp,
} from 'antd';
import { SendOutlined, DeleteOutlined, RedoOutlined } from '@ant-design/icons';

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

  /** ゲーム開始 */
  const startGame = () => {
    const p = generateProblem();
    setGameStarted(true);
    setCurrentProblem(p);
    setMemoryAnswers([]);        // 最初は覚えるだけ
    setScore(0);
    
    // Canvas初期化
    setTimeout(() => {
      if (!canvasRef.current) {
        console.error('Canvas ref is null');
        return;
      }
      
      const canvas = canvasRef.current;
      const cssW = 300, cssH = 300;
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
    if (!canvasRef.current) return;
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
    if (!isDrawing || !canvasRef.current) return;
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
    if (!currentProblem) return;

    // n問たまるまでは“覚えるだけ”
    if (memoryAnswers.length < n) {
      message.info(`ウォームアップ中… あと ${n - memoryAnswers.length} 問は覚えるだけです`);
      setMemoryAnswers((prev) => [...prev, currentProblem.answer]); // 今の答えを積む
      setCurrentProblem(generateProblem());
      clearCanvas();
      return;
    }

    const recognizedDigit = await recognizeDigit();
    if (recognizedDigit === null) return;

    const targetAnswer = memoryAnswers[memoryAnswers.length - n]; // ちょうど n 個前
    if (recognizedDigit === targetAnswer) {
      setScore((s) => s + 1);
      message.success('正解！ +1点');
    } else {
      message.error(`不正解… 正解は ${targetAnswer}`);
    }

    // 今表示していた問題の答えをpush → 次へ
    setMemoryAnswers((prev) => [...prev, currentProblem.answer]);
    setCurrentProblem(generateProblem());
    clearCanvas();
  };

  // useEffectでのCanvas初期化は削除（ゲーム開始時に初期化するため）

  return (
    <Layout className="app-wrap">
      <Header style={{ background: '#0b1220' }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={3} style={{ color: '#e6f4ff', margin: 0 }}>
              Oni-Calc: 記憶力トレーニング
            </Title>
          </Col>
          <Col>
            <Space>
              <Text style={{ color: '#94a3b8' }}>n</Text>
              <InputNumber min={1} max={9} value={n} onChange={(v)=>setN(Number(v)||1)} />
            </Space>
          </Col>
        </Row>
      </Header>

      <Content style={{ padding: 24 }}>
        {!gameStarted ? (
          <Card className="board">
            <Text style={{ color: '#cbd5e1' }}>
              n個前の問題の答えを、手書きで入力するゲームです。最初の n 問は覚えるだけ。
            </Text>
            <Divider />
            <Button type="primary" onClick={startGame}>ゲーム開始</Button>
          </Card>
        ) : (
          <Row gutter={[24, 24]}>
            <Col xs={24} md={16}>
              <Card className="board" styles={{ body: { padding: 20 } }}>
                <Row gutter={[16, 16]}>
                  <Col xs={12} md={6}>
                    <Statistic title="スコア" value={score} valueStyle={{ color: '#f0f9ff' }} />
                  </Col>
                  <Col xs={12} md={18}>
                    <Space>
                      {Array.from({ length: n }).map((_, i) => {
                        // 直近 n 個分のスロット（値は見せない：カンニング防止）
                        const isTarget = i === 0 && memoryAnswers.length >= n;
                        return (
                          <Tag
                            key={i}
                            color={isTarget ? 'green' : 'default'}
                            style={{ padding: '6px 10px', borderRadius: 999 }}
                          >
                            {isTarget ? '●' : '○'}
                          </Tag>
                        );
                      })}
                    </Space>
                  </Col>
                </Row>

                <Divider style={{ borderColor: '#1f2937' }} />

                <Text style={{ color: '#9ca3af' }}>問題</Text>
                <Title level={2} className="problem" style={{ color: '#fff' }}>
                  {currentProblem?.num1} {currentProblem?.operation} {currentProblem?.num2} = ?
                </Title>
                <Text type="secondary">※{n}個前の答えを手書きで入力</Text>

                <Divider />

                <Row gutter={[24, 24]} align="middle">
                  <Col>
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
                  </Col>
                  <Col flex="auto">
                    <Space wrap>
                      <Button icon={<RedoOutlined rotate={180} />} onClick={clearCanvas}>
                        やり直す
                      </Button>
                      <Button danger icon={<DeleteOutlined />} onClick={clearCanvas}>
                        消す
                      </Button>
                      <Button type="primary" icon={<SendOutlined />} onClick={submitAnswer}>
                        送信
                      </Button>
                    </Space>
                  </Col>
                </Row>
              </Card>
            </Col>

            <Col xs={24} md={8}>
              <Card
                className="board"
                title="ヒント"
                styles={{ header: { color: '#e5e7eb' } }}
              >
                <Space direction="vertical">
                  <Text>・最初の <b>{n}</b> 問は覚えるだけ（自動で次へ進みます）</Text>
                  <Text>・答えは 0–9 の一桁のみ</Text>
                </Space>
              </Card>
            </Col>
          </Row>
        )}
      </Content>
    </Layout>
  );
}
