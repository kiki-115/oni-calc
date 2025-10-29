import React from 'react';
import ReactDOM from 'react-dom/client';
import 'antd/dist/reset.css';                  // ← Ant Design のリセットCSS
import App from './App';
import reportWebVitals from './reportWebVitals';

import { ConfigProvider, theme, App as AntdApp } from 'antd';
import jaJP from 'antd/locale/ja_JP';         // ← 日本語ロケール（任意）

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ConfigProvider
      locale={jaJP}
      theme={{
        algorithm: theme.darkAlgorithm,       // ← ダークテーマ（ライトにしたければ削除）
      }}
    >
      <AntdApp>
        <App />
      </AntdApp>
    </ConfigProvider>
  </React.StrictMode>
);

reportWebVitals();
