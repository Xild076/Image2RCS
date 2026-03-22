from duckduckgo_search import DDGS
import pandas as pd
import requests
import os
import time

def download_images(query, limit=100, output_directory='data/inference_images'):
    os.makedirs(output_directory, exist_ok=True)
    paths = []
    
    results = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.images(query, max_results=limit)]
            break
        except Exception as e:
            print(f"Error fetching metadata for {query}: {e}")
            if attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 10
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for {query}. Skipping.")
                return paths
        
    for i, res in enumerate(results):
        try:
            image_url = res['image']
            img_data = requests.get(image_url, timeout=10).content
            filename = f"{query.replace(' ', '_').replace('/', '_')}_{i:03d}.jpg"
            filepath = os.path.join(output_directory, filename)
            
            with open(filepath, 'wb') as handler:
                handler.write(img_data)
            paths.append(filepath)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download image {i}: {e}")
            
    return paths

def fetch_images_of_aircraft(aircraft_csv, limit=100, core_dir='data/images'):
    df = pd.read_csv(aircraft_csv)
    for index, row in df.iterrows():
        name = row['aircraft_name']
        output_dir = core_dir + '/' + row['image_folder']
        print(f"Downloading images for: {name}")
        download_images(name, limit=limit, output_directory=output_dir)
        time.sleep(5)  # Add sleep to prevent hitting rate limits


if __name__ == "__main__":
    fetch_images_of_aircraft('data/aircraft_rcs.csv', limit=100, core_dir='data/images')