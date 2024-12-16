import requests
from PIL import Image
from io import BytesIO

url = 'http://www.boldsystems.org/pics/AUSCL/BIOUG02157-C12+1425493756.jpg'

# Include headers if necessary
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    try:
        im = Image.open(BytesIO(response.content))
        print(im.size)
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Failed to fetch the image. Status code: {response.status_code}")
