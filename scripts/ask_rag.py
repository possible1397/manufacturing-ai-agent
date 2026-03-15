import json
import requests
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "output" / "chroma_db"
COLLECTION_NAME = "maintenance_cases"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b" # 採用輕量級高性能的中文優化模型

def generate_ai_response(query: str, retrieved_cases: str):
    prompt = f"""
你是一個資深 AOI 設備工程師，專精於機台維修與除錯。
請嚴格根據以下歷史維修案例，回答使用者的問題。禁止捏造案例中沒有的資訊。

使用者問題：{query}

歷史案例參考：
{retrieved_cases}

請提供：
1. 可能根因分析
2. 建議處理步驟 (SOP 格式)
3. 參考了哪些案例ID
"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True # 採用流式傳輸，讓回答一個字一個字印出，提升使用者體驗
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        
        print("\n🤖 AI 分析建議：")
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                print(data.get("response", ""), end="", flush=True)
        print("\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[系統提示] 無法連線到本地 Ollama 服務 ({OLLAMA_API_URL})。")
        print(f"請確認 Ollama 已啟動，且已在背景執行 `ollama run {OLLAMA_MODEL}`。")
    except Exception as e:
        print(f"\n[系統錯誤] 發生未知的錯誤: {e}")

def main():
    print("載入 ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("載入 embedding 模型 (用於計算語意相似度)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print(f"\n✅ 系統準備完畢！目前對接的大語言模型：本地 Ollama [{OLLAMA_MODEL}]")
    
    while True:
        query = input("\n請輸入機台異常症狀（輸入 exit 離開）：").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        query_embedding = model.encode([query]).tolist()[0]

        # 檢索 Top 3 最相似的案例
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_texts = []
        # 優化檢索結果的終端機顯示，不再印出落落長的全文
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            doc_id = meta.get('doc_id', 'Unknown')
            category = meta.get('problem_category', '無分類')
            component = meta.get('key_component', '無特定元件')
            
            # 印出給使用者看的簡潔參考
            print(f"🔍 [參考案例 {i}] 相似度: {dist:.4f} | ID: {doc_id} | 分類: {category} | 關鍵元件: {component}")
            
            # 準備餵給 LLM 的上下文 (Context)
            case_context = f"[案例 {i}] ID: {doc_id}\n案例內容: {doc}\n"
            retrieved_texts.append(case_context)

        combined_cases = "\n".join(retrieved_texts)
        
        # 將問題與整理好的歷史案例送往 Ollama 進行邏輯總結
        generate_ai_response(query, combined_cases)

if __name__ == "__main__":
    main()
