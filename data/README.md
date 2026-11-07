# data/

這個資料夾放原始維修日誌，**內容不進版控**。

`scripts/sanitize_data.py` 預期 `Combined_Maintenance_Log_Cleaned.xlsx` 具備以下欄位：

| 欄位 | 說明 |
| --- | --- |
| `Date` | 發生日期 |
| `Client` | 客戶名稱（會被代號化為 `Cust_001`…） |
| `Machine_ID` | 機台編號（會被代號化為 `Machine_001`…） |
| `Problem_Category` | 問題分類 |
| `Key_Component` | 關鍵元件 |
| `Symptom` | 問題現象（自由文字） |
| `Description` / `Result` / `Root_Cause` | 處理過程與結論（自由文字） |

## extra_terms.txt

去識別化只靠欄位對照是不夠的——工程師常把客戶簡稱、聯絡人姓名直接打在
`Symptom` 這類自由欄位裡，那些值不會出現在 `Client` 欄，對照表自然抓不到。

把這些字詞一行一個寫進 `data/extra_terms.txt`（同樣不進版控），
`sanitize_data.py` 會一併遮蔽。Email 與電話則由正則處理，不需手動列舉。

寫檔前腳本會對全表稽核一次，偵測到漏網就中止並保留原檔。
