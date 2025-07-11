import os
import requests

# URL of a sample JPEG image (public domain)
url = 'https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png'

# Ensure the assets directory exists
os.makedirs('assets', exist_ok=True)

# Download the image
response = requests.get(url)
if response.status_code == 200:
    with open('assets/sample.jpg', 'wb') as f:
        f.write(response.content)
    print('Sample image downloaded and saved as assets/sample.jpg')
else:
    print('Failed to download image. Status code:', response.status_code) 