import pandas as pd
from pathlib import Path
import shutil
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
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
        return text

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(mask_text)

    # Save the sanitized payload over the original file
    df.to_excel(EXCEL_FILE, index=False)
    
    print(f"成功去識別化資料。")
    print(f"- 替換了 {len(client_map)} 個客戶名稱")
    print(f"- 替換了 {len(machine_map)} 個機台號碼")
    print(f"- 已直接覆蓋原始檔案供後續 RAG 使用 (原檔備份於 Backup.xlsx)")

if __name__ == "__main__":
    sanitize_data()
