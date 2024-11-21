import requests
import os

url = 'https://www.dropbox.com/scl/fi/mao3s9g88my3j7mgsvjcj/DD_Image.h5?rlkey=5y1u7ncxxjbopr7gt6qec8qih&st=nevrql26&dl=1'
file_path = './DD_Image.h5'

def download_model():
    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)