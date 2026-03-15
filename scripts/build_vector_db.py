import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSONL = BASE_DIR / "output" / "cases.jsonl"
CHROMA_DIR = BASE_DIR / "output" / "chroma_db"
COLLECTION_NAME = "maintenance_cases"


def load_cases(jsonl_path: Path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"找不到檔案：{INPUT_JSONL}")

    print("載入案例資料...")
    records = load_cases(INPUT_JSONL)

    print("載入 embedding 模型...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("初始化 ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []

    for record in records:
        ids.append(record["doc_id"])
        documents.append(record["text"])
        metadatas.append(record["metadata"])

    print("建立 embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # 若重跑，先刪掉舊資料再重建
    existing = collection.count()
    if existing > 0:
        print(f"偵測到既有資料 {existing} 筆，將刪除後重建...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("寫入 ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("完成！")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"筆數: {collection.count()}")
    print(f"路徑: {CHROMA_DIR}")


if __name__ == "__main__":
    main()