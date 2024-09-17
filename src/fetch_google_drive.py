# Import the necessary libraries for Google Drive handling
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import os
import io
from test3.ipynb import analyze_images

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = '.json'
SUBMISSION_FOLDER_ID = "1o0w7mOrhZh5tCuaVXnpBa8195J5iUBz2"
DATA_PATH = 'submissions'
TRAIN_RATIO = 4

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# Function to download a folder and process images
def download_folder_and_analyze(folder_id, save_path):
    request = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id,name,mimeType)").execute()
    files = request.get('files', [])
    current_dir = save_path
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_mime_type = file.get('mimeType', '')
        
        if file_mime_type == 'application/vnd.google-apps.folder':
            os.makedirs(current_dir + "/" + file_name, exist_ok=True)
            download_folder_and_analyze(file_id, current_dir + "/" + file_name)
        else:
            if file_mime_type == 'image/png':  # Only process PNG files
                local_path = current_dir + "/" + file_name
                download_file(file_id, local_path)
                # Call Vision API to analyze the image
                analyze_images([local_path])

# Function to download a specific file
def download_file(file_id, save_path):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        with open(save_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Downloading {file_id}: {int(status.progress() * 100)}%")
    except HttpError as error:
        print(f"An error occurred: {error} \nFile {file_id} failed")
        return None

# Download the submissions folder and process
request = drive_service.files().list(q=f"'{SUBMISSION_FOLDER_ID}' in parents", fields="files(id,name)").execute()
files = request.get('files', [])

for file in files:
    file_id = file['id']
    file_name = file['name']
    # Replace logic with your specific folder handling
    download_folder_and_analyze(file_id, DATA_PATH)
