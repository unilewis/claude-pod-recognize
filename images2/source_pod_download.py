import os
import requests
import mimetypes
from pathlib import Path

def download_images(source_file, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    with open(source_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(urls)} URLs. Starting download...")
    
    for i, url in enumerate(urls, 1):
        try:
            filename_base = f"pod_{i:03d}"
            
            # Use streaming to handle large files and get headers first
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Determine extension
            content_type = response.headers.get('content-type')
            extension = mimetypes.guess_extension(content_type)
            if not extension:
                # Fallback to URL or default to .jpg
                if '.jpeg' in url.lower(): extension = '.jpeg'
                elif '.png' in url.lower(): extension = '.png'
                else: extension = '.jpg'
            
            target_path = os.path.join(target_dir, f"{filename_base}{extension}")
            
            with open(target_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
            
            print(f"[{i}/{len(urls)}] Downloaded {url} -> {os.path.basename(target_path)}")
            
        except Exception as e:
            print(f"[{i}/{len(urls)}] Failed to download {url}: {e}")

if __name__ == "__main__":
    SOURCE = "./source_pod.txt"
    TARGET = "./"
    download_images(SOURCE, TARGET)
