import requests
import os

base_folder = "dataset"
os.makedirs(base_folder, exist_ok=True)

with open("urls.txt", "r", encoding="utf-8") as f:
    urls = f.read().splitlines()

for url in urls:
    if not url.strip():
        continue

    path_part = url.split("/V2/葡萄病害600X400/")[-1]  
    folder_name, file_name = path_part.split("/", 1)
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    r = requests.get(url)
    with open(os.path.join(folder_path, file_name.split("&")[0]), "wb") as f:
        f.write(r.content)

    print(f"Downloaded {folder_name}/{file_name.split('&')[0]}")
