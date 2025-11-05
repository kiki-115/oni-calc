import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import {
  Layout, Row, Col, Card, Space, Typography,
  Button, Statistic, Tag, InputNumber, Divider,
  App as AntdApp, Modal,
} from 'antd';
import { SendOutlined, DeleteOutlined, UpOutlined, DownOutlined, HomeOutlined, QuestionCircleOutlined, RedoOutlined } from '@ant-design/icons';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

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
  const { message } = AntdApp.useApp();

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentProblem, setCurrentProblem] = useState(null);
  const [memoryAnswers, setMemoryAnswers] = useState([]);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameFinished, setGameFinished] = useState(false);
  const [n, setN] = useState(3);
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  
  const [problemHistory, setProblemHistory] = useState([]);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [showJudgment, setShowJudgment] = useState(false);
  const [pendingJudgment, setPendingJudgment] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hintModalVisible, setHintModalVisible] = useState(false);
  
  const [startTime, setStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [answeredCount, setAnsweredCount] = useState(0);
  const [wrongCount, setWrongCount] = useState(0);
  const [finalScore, setFinalScore] = useState(null);
  const [finalTime, setFinalTime] = useState(null);
  const [answerHistory, setAnswerHistory] = useState([]);

  const startGame = () => {
    const p = generateProblem();
    setGameStarted(true);
    setGameFinished(false);
    setCurrentProblem(p);
    setMemoryAnswers([]);
    setProblemHistory([p]);
    setQuestionNumber(1);
    setShowJudgment(false);
    setPendingJudgment(null);
    setIsSubmitting(false);
    setStartTime(null);
    setElapsedTime(0);
    setAnsweredCount(0);
    setWrongCount(0);
    setFinalScore(null);
    setAnswerHistory([]);
    
    // Canvas初期化
    setTimeout(() => {
      if (!canvasRef.current) {
        console.error('Canvas ref is null');
        return;
      }
      
      const canvas = canvasRef.current;
      const cssW = 400, cssH = 400;
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
      canvas.style.width = cssW + 'px';
      canvas.style.height = cssH + 'px';
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }, 100);
  };

  const restartGame = () => {
    const p = generateProblem();
    setCurrentProblem(p);
    setMemoryAnswers([]);
    setProblemHistory([p]);
    setQuestionNumber(1);
    setShowJudgment(false);
    setPendingJudgment(null);
    setIsSubmitting(false);
    setStartTime(null);
    setElapsedTime(0);
    setAnsweredCount(0);
    setWrongCount(0);
    setGameFinished(false);
    setFinalScore(null);
    setAnswerHistory([]);
    
    setTimeout(() => {
      if (!canvasRef.current) {
        console.error('Canvas ref is null');
        return;
      }
      
      const canvas = canvasRef.current;
      const cssW = 400, cssH = 400;
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
      canvas.style.width = cssW + 'px';
      canvas.style.height = cssH + 'px';
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }, 100);
    
    message.info('ゲームをリセットしました');
  };

  const getCanvasPoint = (e) => {
    // 座標変換
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    if (!canvasRef.current || memoryAnswers.length < n) return;
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.lineWidth = 10 * dpr;
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
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const recognizeDigit = async () => {
    if (!canvasRef.current) return null;
    const canvas = canvasRef.current;

    // API送信
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

  const submitAnswer = async () => {
    if (!currentProblem || memoryAnswers.length < n || isSubmitting || showJudgment || pendingJudgment) return;

    setIsSubmitting(true);

    const recognizedDigit = await recognizeDigit();
    if (recognizedDigit === null) {
      setIsSubmitting(false);
      return;
    }

    // n個前の問題の答えと比較
    const targetProblem = problemHistory[0];
    const targetAnswer = memoryAnswers[memoryAnswers.length - n];
    
    const newHistory = [...problemHistory];
    newHistory[0] = {
      ...problemHistory[0],
      recognizedAnswer: recognizedDigit,
      isCorrect: recognizedDigit === targetAnswer
    };
    setProblemHistory(newHistory);
    
    setPendingJudgment({ 
      recognizedAnswer: recognizedDigit, 
      isCorrect: recognizedDigit === targetAnswer 
    });
    
    setTimeout(() => {
      setShowJudgment(true);
      
      const isCorrect = recognizedDigit === targetAnswer;
      
      setAnswerHistory((prev) => [...prev, isCorrect ? 'o' : 'x']);
      
      if (!isCorrect) {
        setWrongCount((prev) => prev + 1);
      }
      
      setAnsweredCount((prev) => prev + 1);
      
      // 40問で終了
      const currentAnswered = answeredCount + 1;
      if (currentAnswered >= 40) {
        const localFinalTime = (Date.now() - startTime) / 1000;
        setFinalTime(localFinalTime);
        const finalWrongCount = wrongCount + (isCorrect ? 0 : 1);
        const calculatedScore = Number((localFinalTime + finalWrongCount * 8).toFixed(1));
        setFinalScore(calculatedScore);
        setGameFinished(true);
        setIsSubmitting(false);
        clearCanvas();
        return;
      }
      
      // 次の問題へ
      setTimeout(() => {
        setPendingJudgment(null);
        setShowJudgment(false);
        setIsSubmitting(false);
        
        const newProblem = generateProblem();
        setMemoryAnswers((prev) => [...prev, currentProblem.answer]);
        setCurrentProblem(newProblem);
        setQuestionNumber((prev) => prev + 1);
        setProblemHistory((prev) => {
          const updated = [...prev];
          updated.shift();
          updated.push(newProblem);
          return updated;
        });
        clearCanvas();
      }, 300);
    }, 200);
  };

  // タイマー開始
  useEffect(() => {
    if (!gameStarted || gameFinished || memoryAnswers.length !== n || startTime !== null) return;
    
    setStartTime(Date.now());
  }, [gameStarted, gameFinished, memoryAnswers.length, n, startTime]);

  // 経過時間更新
  useEffect(() => {
    if (!gameStarted || gameFinished || !startTime) return;
    
    const timer = setInterval(() => {
      if (startTime) {
        setElapsedTime((Date.now() - startTime) / 1000);
      }
    }, 100);
    
    return () => clearInterval(timer);
  }, [gameStarted, gameFinished, startTime]);

  // 記憶フェーズ（自動問題切り替え）
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
      <Header style={{ 
        background: 'linear-gradient(135deg, #0b1220 0%, #1e293b 50%, #0b1220 100%)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        paddingTop: '16px',
        height: 'auto',
        minHeight: '64px'
      }}>
        <Row align="bottom" justify="space-between" style={{ paddingBottom: '8px' }}>
          <Col>
            <Title 
              level={2} 
              style={{ 
                margin: 0,
                paddingBottom: '4px',
                background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #f97316 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                fontSize: '28px',
                fontWeight: 700,
                letterSpacing: '0.05em',
                textShadow: '0 2px 8px rgba(251, 191, 36, 0.3)',
                fontFamily: '"Hiragino Sans", "ヒラギノ角ゴ ProN", "Hiragino Kaku Gothic ProN", "游ゴシック", "Yu Gothic", "メイリオ", Meiryo, sans-serif'
              }}
            >
              鬼計算 : 記憶力トレーニング
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
        ) : gameFinished ? (
          <Card className="board" styles={{ body: { padding: 40 } }}>
            <Space direction="vertical" size="large" style={{ width: '100%', textAlign: 'center' }}>
              <Title level={2} style={{ color: '#fff' }}>お疲れ様でした！</Title>
              <div style={{ padding: '24px', background: '#1f2937', borderRadius: 8 }}>
                <Text style={{ color: '#9ca3af', fontSize: 16, display: 'block', marginBottom: 8 }}>
                  解答結果
                </Text>
                <div style={{ 
                  fontFamily: 'monospace', 
                  fontSize: 20, 
                  letterSpacing: '3px',
                  color: '#fff',
                  wordBreak: 'break-all',
                  textAlign: 'center'
                }}>
                  {answerHistory.map((result, idx) => (
                    <span key={idx} style={{ color: result === 'o' ? '#22c55e' : '#ef4444' }}>
                      {result}
                    </span>
                  ))}
                </div>
              </div>
              <div style={{ padding: '24px', background: '#1f2937', borderRadius: 8 }}>
                <Text style={{ color: '#9ca3af', fontSize: 16, display: 'block', marginBottom: 8 }}>
                  かかった時間
                </Text>
                <Title level={1} style={{ color: '#fff', margin: 0 }}>
                  {typeof finalTime === 'number' ? finalTime.toFixed(1) : finalTime}秒
                </Title>
              </div>
              <div style={{ padding: '24px', background: '#1f2937', borderRadius: 8 }}>
                <Text style={{ color: '#9ca3af', fontSize: 16, display: 'block', marginBottom: 8 }}>
                  間違い数
                </Text>
                <Title level={1} style={{ color: '#fff', margin: 0 }}>
                  {wrongCount}問
                </Title>
              </div>
              <div style={{ padding: '24px', background: '#0f172a', borderRadius: 8, border: '2px solid #fbbf24' }}>
                <Text style={{ color: '#fbbf24', fontSize: 20, display: 'block', marginBottom: 8 }}>
                  最終スコア（時間 + 間違い×8秒）
                </Text>
                <Title level={1} style={{ color: '#fbbf24', margin: 0 }}>
                  {finalScore !== null ? finalScore.toFixed(1) : 0}秒
                </Title>
              </div>
              <Button type="primary" size="large" onClick={() => { setGameStarted(false); setGameFinished(false); }}>
                ホームに戻る
              </Button>
            </Space>
          </Card>
        ) : (
          <Row gutter={[24, 24]}>
            <Col xs={24} md={12}>
              <Card className="board" styles={{ body: { padding: 20 } }}>
                <Row gutter={[16, 16]}>
                  <Col xs={12} md={6}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Button 
                          type="text"
                          icon={<QuestionCircleOutlined />}
                          onClick={() => setHintModalVisible(true)}
                          style={{ color: '#94a3b8' }}
                        />
                        <Statistic title="経過時間" value={`${typeof elapsedTime === 'number' ? elapsedTime.toFixed(1) : elapsedTime}秒`} valueStyle={{ color: '#f0f9ff', fontSize: 24 }} />
                      </div>
                      <Text style={{ color: '#94a3b8', fontSize: 14 }}>残り: {40 - answeredCount}問</Text>
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
                        icon={<RedoOutlined />} 
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

                <div style={{ marginBottom: 24 }}>
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
                              textAlign: 'center',
                              marginLeft: '0.5em'
                            }}>
                              {pendingJudgment.recognizedAnswer}
                            </span>
                          </>
                        ) : (
                          <>
                            ? <span style={{ fontSize: '0.7em' }}>?</span> ? = 
                            <span style={{
                              display: 'inline-block',
                              width: '30px',
                              height: '30px',
                              background: 'rgba(255, 255, 255, 0.1)',
                              borderRadius: '4px',
                              marginLeft: '0.5em'
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
                        style={{ touchAction: 'none' }}
                      />
                    </div>
                  </div>
                  <Space wrap style={{ width: '100%', justifyContent: 'center' }}>
                    <Button 
                      danger 
                      icon={<DeleteOutlined />} 
                      onClick={clearCanvas}
                      disabled={memoryAnswers.length < n || isSubmitting || showJudgment || pendingJudgment}
                    >
                      消す
                    </Button>
                    <Button 
                      type="primary" 
                      icon={<SendOutlined />} 
                      onClick={submitAnswer}
                      disabled={memoryAnswers.length < n || isSubmitting || showJudgment || pendingJudgment}
                      style={{ opacity: (memoryAnswers.length < n || isSubmitting || showJudgment || pendingJudgment) ? 0.5 : 1 }}
                    >
                      送信
                    </Button>
                  </Space>
                </Space>
              </Card>
            </Col>
          </Row>
        )}

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
