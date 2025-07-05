import zipfile
import os

zip_path = 'data/raw/complaints.csv.zip'  # your ZIP file path
extract_to = 'data/raw/complaints/'       # extraction folder

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Extracted files to {extract_to}")
