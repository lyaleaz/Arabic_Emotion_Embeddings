
import pandas as pd

file_path = "Arabic_Emotion_Plutchik8_datasets.xlsx"

excel = pd.ExcelFile(file_path)

print("WORKSHEETS:")
print(excel.sheet_names)

for sheet in excel.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    print("\n==============================")
    print("Sheet:", sheet)
    print("Number of rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
