import pandas as pd
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "Combined_Maintenance_Log_Cleaned.xlsx"
OUTPUT_JSONL = BASE_DIR / "output" / "cases.jsonl"
OUTPUT_CSV = BASE_DIR / "output" / "cases_for_rag.csv"


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def safe_get(row, col_name: str) -> str:
    return clean_text(row[col_name]) if col_name in row else ""


def build_full_content(symptom: str, solution: str, root_cause: str, full_content: str) -> str:
    if full_content:
        return full_content

    parts = []
    if symptom:
        parts.append(f"症狀：{symptom}")
    if solution:
        parts.append(f"處理方式：{solution}")
    if root_cause:
        parts.append(f"根因：{root_cause}")

    return "；".join(parts)


def build_case_text(row, index: int):
    doc_id = safe_get(row, "Doc_ID") or f"CASE_{index:04d}"
    date = safe_get(row, "Date")
    client = safe_get(row, "Client")
    machine_id = safe_get(row, "Machine_ID")
    problem_category = safe_get(row, "Problem_Category")
    key_component = safe_get(row, "Key_Component")
    symptom = safe_get(row, "Symptom")
    solution = safe_get(row, "Solution")
    root_cause = safe_get(row, "Root_Cause")
    full_content = safe_get(row, "Full_Content")
    source_file = safe_get(row, "Source_File")
    source_sheet = safe_get(row, "Source_Sheet")

    full_content = build_full_content(symptom, solution, root_cause, full_content)

    lines = ["【維修案例】", f"案例ID：{doc_id}"]
    if date:
        lines.append(f"日期：{date}")
    if client:
        lines.append(f"客戶：{client}")
    if machine_id:
        lines.append(f"機台編號：{machine_id}")
    if problem_category:
        lines.append(f"問題分類：{problem_category}")
    if key_component:
        lines.append(f"關鍵元件：{key_component}")
    if symptom:
        lines.append(f"症狀：{symptom}")
    if solution:
        lines.append(f"處理方式：{solution}")
    if root_cause:
        lines.append(f"根因：{root_cause}")
    if full_content:
        lines.append(f"完整內容：{full_content}")
    if source_file:
        lines.append(f"來源檔案：{source_file}")
    if source_sheet:
        lines.append(f"來源工作表：{source_sheet}")

    text = "\n".join(lines)

    metadata = {
        "doc_id": doc_id,
        "date": date,
        "client": client,
        "machine_id": machine_id,
        "problem_category": problem_category,
        "key_component": key_component,
        "source_file": source_file,
        "source_sheet": source_sheet,
    }

    return text, metadata


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_FILE)

    jsonl_records = []
    csv_records = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        text, metadata = build_case_text(row, idx)

        jsonl_records.append({
            "doc_id": metadata["doc_id"],
            "text": text,
            "metadata": metadata
        })

        csv_records.append({
            "doc_id": metadata["doc_id"],
            "text": text,
            **metadata
        })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    pd.DataFrame(csv_records).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("完成輸出：")
    print(f"- {OUTPUT_JSONL}")
    print(f"- {OUTPUT_CSV}")
    print(f"- 總筆數：{len(jsonl_records)}")


if __name__ == "__main__":
    main()