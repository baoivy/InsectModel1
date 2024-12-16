import json
from PIL import Image, UnidentifiedImageError
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# Filepath to metadata
metadata_filepath = 'Insect-1M-v1.json'

# Directory to save downloaded images
output_dir = 'insect_images1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load metadata
with open(metadata_filepath, 'r') as f:
    metadata = json.load(f)

insect_records = metadata['insect_records']

# Function to download a single image
def download_image(item, session):
    img_link = item['image_url']
    img_id = item.get('id', 'unknown')  # Fallback if 'id' is missing

    # Build the file path
    img_path = os.path.join(output_dir, f'insect{img_id}.jpg')

    # Skip download if the file already exists
    if os.path.exists(img_path):
        return f"Image {img_id} already exists."

    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Fetch the image
        response = session.get(img_link, headers=headers, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses

        # Process and save the image
        img = Image.open(BytesIO(response.content))  # Re-open for saving
        img.save(img_path)
        return f"Downloaded image {img_id}."
    except UnidentifiedImageError:
        return f"Failed to identify image {img_id} from {img_link}."
    except Exception as e:
        return f"Failed to download image {img_id} from {img_link}: {e}"

# Multithreaded download
def download_images_multithreaded(records, max_threads=10):
    with requests.Session() as session:
        with ThreadPoolExecutor(max_threads) as executor:
            # Submit tasks
            futures = [executor.submit(download_image, item, session) for i, item in enumerate(records) if i >= 114285]

            # Process results with a progress bar
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                print(result)

# Start downloading images
download_images_multithreaded(insect_records, max_threads=10)
