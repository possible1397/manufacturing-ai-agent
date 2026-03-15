import streamlit as st
import json
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 配置區 ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "output" / "chroma_db"
COLLECTION_NAME = "maintenance_cases"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

# --- 頁面設定 ---
st.set_page_config(page_title="AOI 維修 AI 助理", page_icon="🤖", layout="wide")
st.title("🤖 AOI 製造業維修助理 (RAG + Ollama)")
st.markdown("輸入機台異常症狀，AI 將從歷史維修紀錄中為您尋找相似案例，並整理出排查 SOP。")

# --- 載入模型 (利用 st.cache_resource 避免重複載入) ---
@st.cache_resource(show_spinner="載入向量模型與資料庫中...")
def load_resources():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return collection, model

try:
    collection, model = load_resources()
except Exception as e:
    st.error(f"系統初始化失敗: {e}")
    st.stop()


def get_ollama_response(prompt):
    """呼叫 Ollama API 並回傳 Stream (產生器) 供 st.write_stream 使用"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if "response" in data:
                    yield data["response"]
    except Exception as e:
        yield f"\n\n🚨 無法連線至 Ollama ({OLLAMA_MODEL})，請確認系統已啟動: {str(e)}"

# --- UI 聊天區塊 ---
# 儲存對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 若是助手的回答，一併顯示當下參考的文獻
        if msg["role"] == "assistant" and "references" in msg:
            with st.expander("🔍 查看參考歷史案例"):
                st.markdown(msg["references"])

# 處理使用者新輸入
if prompt := st.chat_input("請描述機台異常狀況 (例如：相機抓不到影像)"):
    # 1. 顯示使用者訊息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 開始檢索資料 (RAG - Retrieval)
    with st.spinner("🔍 正在從歷史資料庫檢索相似案例..."):
        query_embedding = model.encode([prompt]).tolist()[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_texts = []
        reference_markdown = ""
        
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            doc_id = meta.get('doc_id', 'Unknown')
            category = meta.get('problem_category', '無分類')
            component = meta.get('key_component', '無特定元件')
            
            # 準備餵給 LLM 的上下文 (Context)
            case_context = f"[案例 {i}] ID: {doc_id}\n案例內容: {doc}\n"
            retrieved_texts.append(case_context)
            
            # 準備顯示給使用者看的 UI 參考
            reference_markdown += f"**[{i}] 相似度 {dist:.2f}** | 分類: `{category}` | 關鍵元件: `{component}`  \n"
            reference_markdown += f"> *ID: {doc_id}*  \n\n"
            
        combined_cases = "\n".join(retrieved_texts)

    # 3. 組裝 Prompt 並呼叫 LLM (Generative)
    llm_prompt = f"""
你是一個資深 AOI 設備工程師，專精於機台維修與除錯。
請嚴格根據以下歷史維修案例，回答使用者的問題。禁止捏造案例中沒有的資訊。

使用者問題：{prompt}

歷史案例參考：
{combined_cases}

請提供：
1. 可能根因分析
2. 建議處理步驟 (SOP 格式)
3. 參考了哪些案例ID
"""

    # 4. 顯示 LLM 回答
    with st.chat_message("assistant"):
        # 首先用流式顯示文字
        response = st.write_stream(get_ollama_response(llm_prompt))
        
        # 文字輸出完畢後，用展開框附上參考文獻
        with st.expander("🔍 查看參考歷史案例", expanded=True):
            st.markdown(reference_markdown)

    # 紀錄到 Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "references": reference_markdown
    })
