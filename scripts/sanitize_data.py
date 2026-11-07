import pandas as pd
from pathlib import Path
import shutil
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- 通用 PII 遮蔽 ---------------------------------------------------------
# 下方的 client_map / machine_map 只能遮蔽「出現在 Client / Machine_ID 欄位」的值。
# 工程師常把客戶名、聯絡人、Email 與電話直接打在問題描述裡，那些值從未進過
# 對照表，因此完全不會被取代。以下規則不依賴對照表，獨立再掃一次。
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
MOBILE_RE = re.compile(r"(?<![\d-])09\d{2}[-\s]?\d{3}[-\s]?\d{3}(?![\d-])")
TEL_RE = re.compile(r"\(0\d{1,2}\)\s?\d{3,4}[-\s]?\d{4}")

# 自由欄位中仍需遮蔽的字詞（客戶簡稱、人名…）。
# 這份清單本身就是敏感資訊，放在不進版控的 data/extra_terms.txt，一行一個詞。
EXTRA_TERMS_FILE = DATA_DIR / "extra_terms.txt"


def load_extra_terms():
    if not EXTRA_TERMS_FILE.exists():
        return []
    with open(EXTRA_TERMS_FILE, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def scrub_pii(text: str, extra_terms=()) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = MOBILE_RE.sub("[PHONE]", text)
    text = TEL_RE.sub("[PHONE]", text)
    for term in sorted((t for t in extra_terms if t), key=len, reverse=True):
        text = text.replace(term, "[REDACTED]")
    return text


def audit(df, extra_terms=()):
    """輸出前的最後一道檢查，回傳 [(類別, 範例)]。"""
    joined = chr(10).join(
        df[c].astype(str).str.cat(sep=chr(10))
        for c in df.columns
        if df[c].dtype == object
    )
    found = []
    for kind, pat in (("Email", EMAIL_RE), ("電話", MOBILE_RE), ("電話", TEL_RE)):
        m = pat.search(joined)
        if m:
            found.append((kind, m.group(0)))
    for term in extra_terms:
        if term and term in joined:
            found.append(("指定字詞", term))
    return found
# ---------------------------------------------------------------------------
EXCEL_FILE = DATA_DIR / "Combined_Maintenance_Log_Cleaned.xlsx"
BACKUP_FILE = DATA_DIR / "Combined_Maintenance_Log_Cleaned_Backup.xlsx"

def sanitize_data():
    if not EXCEL_FILE.exists():
        print(f"找不到檔案: {EXCEL_FILE}")
        return

    # Backup the original file just in case
    if not BACKUP_FILE.exists():
        shutil.copy2(EXCEL_FILE, BACKUP_FILE)
        print(f"已建立原文備份檔: {BACKUP_FILE}")

    df = pd.read_excel(EXCEL_FILE)
    extra_terms = load_extra_terms()

    # 1. Identifier Mappings
    # Safely get unique, non-null clients and machine IDs
    unique_clients = df['Client'].dropna().unique()
    unique_machines = df['Machine_ID'].dropna().unique()

    client_map = {}
    for i, client in enumerate(unique_clients, 1):
        if str(client).strip() == "":
            continue
        # We might want to keep "廠內" (internal) unmasked, but to be sure we mask everything.
        if "廠內" in str(client):
            client_map[str(client)] = "Internal_Dept"
        else:
            client_map[str(client)] = f"Cust_{i:03d}"

    machine_map = {}
    for i, machine in enumerate(unique_machines, 1):
        if str(machine).strip() == "":
            continue
        machine_map[str(machine)] = f"Machine_{i:03d}"

    # 2. Replace Values in Columns
    if 'Client' in df.columns:
        df['Client'] = df['Client'].map(lambda x: client_map.get(str(x), x) if pd.notna(x) else x)

    if 'Machine_ID' in df.columns:
        df['Machine_ID'] = df['Machine_ID'].map(lambda x: machine_map.get(str(x), x) if pd.notna(x) else x)

    # 3. Textual Replacement inside Content Columns
    def mask_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        # Sort keys by length descending to prevent partial replacements (e.g., replacing 'ABC' before 'ABC_Corp')
        all_maps = {**client_map, **machine_map}
        sorted_keys = sorted(all_maps.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            if key in text:
                text = text.replace(key, all_maps[key])
        return scrub_pii(text, extra_terms)

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(mask_text)

    # 寫檔前稽核：確認沒有漏網的聯絡資訊或指定字詞
    leaks = audit(df, extra_terms)
    if leaks:
        for kind, sample in leaks:
            print(f"[!] 偵測到未遮蔽的{kind}：{sample}")
        raise SystemExit(
            f"中斷：請把漏網的字詞加進 {EXTRA_TERMS_FILE} 後重跑，原始檔未被覆蓋。"
        )

    # Save the sanitized payload over the original file
    df.to_excel(EXCEL_FILE, index=False)
    
    print(f"成功去識別化資料。")
    print(f"- 替換了 {len(client_map)} 個客戶名稱")
    print(f"- 替換了 {len(machine_map)} 個機台號碼")
    print(f"- 已直接覆蓋原始檔案供後續 RAG 使用 (原檔備份於 Backup.xlsx)")

if __name__ == "__main__":
    sanitize_data()
