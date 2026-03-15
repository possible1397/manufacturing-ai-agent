from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "output" / "chroma_db"
COLLECTION_NAME = "maintenance_cases"


def main():
    print("載入 ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("載入 embedding 模型...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    while True:
        query = input("\n請輸入問題（輸入 exit 離開）：").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        query_embedding = model.encode([query]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print("\n=== 查詢結果 ===")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            print(f"\n--- Top {i} ---")
            print(f"相似距離: {dist}")
            print(f"案例ID: {meta.get('doc_id', '')}")
            print(f"問題分類: {meta.get('problem_category', '')}")
            print(f"關鍵元件: {meta.get('key_component', '')}")
            print(doc[:800])  # 先顯示前 800 字


if __name__ == "__main__":
    main()