# AOI 製造業維修助理 (RAG System) 架構全貌

本系統基於 Retrieval-Augmented Generation (RAG) 架構，結合本地端 ChromaDB 向量資料庫與 Ollama 開源大型語言模型，打造完全不需連網、無機密外洩風險的維修知識輔助系統。

整體架構可分為「離線知識庫建置」與「本地即時問答推論」兩大階段：

```text
[ manufacturing-ai-agent 專案 ]
│
├── 📂 第一階段：離線知識庫建置 (Offline Knowledge Construction)
│   │
│   ├── 1. 準備原始數據 (Data Source)
│   │   ├── 📄 data/Combined_Maintenance_Log_Cleaned.xlsx
│   │   └── 💡 第一線設備工程師長期累積的維修日誌 (Excel / CSV)
│   │
│   ├── 2. 數據清洗與文本重組 (Data Preprocessing)
│   │   ├── ⚙️ scripts/prepare_rag_cases.py
│   │   ├── 🧹 清理空值、將結構化表格組合成人類易讀的段落 (Text)
│   │   ├── 🏷️ 提取分類標籤，作為詮釋資料 (Metadata)
│   │   └── 輸出 ➔ 📄 output/cases.jsonl, cases_for_rag.csv
│   │
│   └── 3. 語意向量化與尋機資料庫建置 (Embedding & Vector DB)
│       ├── ⚙️ scripts/build_vector_db.py
│       ├── 🧠 Embedding 模型：`paraphrase-multilingual-MiniLM-L12-v2`
│       │   └── 💡 選型原因：輕量、速度快，且原生支援多國語言語意轉換（包含繁體中文）。
│       └── 輸出 ➔ 📁 output/chroma_db (本地端 ChromaDB 向量圖書館)
│
└── 📂 第二階段：本地即時問答推論 (Local Interactive Q&A Inference)
    │
    ├── 4. 使用者查詢 (User Query)
    │   ├── 💻 app.py (Streamlit 網頁前端)
    │   └── 👤 使用者輸入異常狀況 (例如：「相機抓不到影像」)
    │
    ├── 5. 語意相似度檢索 (Retrieval - **R** of RAG)
    │   ├── 🧠 app.py 裡的 ChromaDB 查詢邏輯
    │   ├── 📐 將這句 User Query 也轉成數學向量
    │   ├── 🔍 進入 output/chroma_db 比對最近的「前 3 筆歷史案例」
    │   └── 輸出 ➔ 📃 3 篇相似案例的內容 (做為上下文 Context)
    │
    └── 6. AI 增強總結生成 (Augmented Generation - **AG** of RAG)
        ├── 🤖 本地端大語言模型 (LLM)：Ollama `qwen2.5:1.5b`
        │   └── 💡 選型原因：在無獨立顯卡 (內顯/純 CPU) 的硬體限制下，1.5B 參數量的 Qwen2.5 能提供流暢的打字速度，且中文理解能力不僅優於多數同級模型，更能在資源受限的筆電中擔任稱職的總結代理。
        ├── 📦 將「前 3 筆歷史案例(Context)」+「使用者查詢(Query)」+「系統提示詞(Prompt)」打包
        ├── 📝 Ollama 根據打包內容進行閱讀理解與邏輯總結
        └── 輸出 ➔ 💡 將「可能原因」與「排查 SOP」回傳至 Streamlit 畫面，完成互動！

## ✅ 常見問題與除錯指南 (Troubleshooting)
當您發現 AI 回答「不精確」或「答非所問」時，問題通常出在以下兩個環節：

### 1. 檢索環節失敗 (找錯參考案例)
如果展開網頁底部的 `[查看參考歷史案例]`，發現找出來的文章與提問完全無關，這通常跟大模型無關，而是檢索端出問題：
- **Excel 源頭資料品質不良**：當工程師在紀錄裡只寫了「機台 NG」、「不行了」，缺乏如「相機、讀取失敗」等關鍵字，Embedding 模型就無法將未來的提問與該紀錄產生對應。
- **Embedding 模型語意理解受限**：如果您大量使用特有的廠內簡稱，您可以考慮未來把模型替換為更強大的 `bge-m3` 等中文特化向量模型，或是導入 Metadata 的篩選功能縮小搜尋範圍。

### 2. 生成環節失敗 (看對文章但畫錯重點)
如果 `[查看參考歷史案例]` 找出來的文章非常正確，但 AI 總結出來的 SOP 卻在講廢話或捏造事實，這就是大模型端的問題：
- **模型腦容量限制**：`1.5b` 的小模型對於過於複雜或冗長的邏輯推演容易產生「幻覺」。
- **解決方法 (Prompt Engineering)**：需在 `app.py` 的程式碼中加強系統指令，例如強制規定：「如果歷史案例沒有提到解法，請回答『歷史案例中無此解法』，嚴禁自行推測！」。
```

## 維運說明
- **資料更新**：當有新的維修紀錄需要匯入時，只需重新執行 `scripts/prepare_rag_cases.py` 與 `scripts/build_vector_db.py` 即可更新 ChromaDB。不需要重啟或修改主程式。
- **UI 服務**：網頁介面透過 `streamlit run app.py` 啟動，並即時向本地的 `localhost:11434` (Ollama) 發出請求。
