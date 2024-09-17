from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

SKIP_EXIST_DIR = 1
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = 'service_account.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# This is the ID of a test drive I made
SUBMISSION_FOLDER_ID = "1-SFFGspWjvHu0JKsqoWUinpdZtRAqs1v"

# Path to where the test and training data will be created if it does not exist
DATA_PATH = r'\submissions'

# How many training images for each test image
TRAIN_RATIO = 4
test_path = os.path.join(DATA_PATH, 'test')
train_path = os.path.join(DATA_PATH, 'train')

train_txt_path = os.path.join(train_path, 'train.txt')
test_txt_path = os.path.join(test_path, 'test.txt')

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

drive_service = build('drive', 'v3', credentials=credentials)

# Read train.txt and test.txt
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    train_num = int(lines[0])
    train = [line.strip() for line in lines[1:]]
logging.info(f"Train assignments: {train}")

with open(test_txt_path, 'r') as f:
    lines = f.readlines()
    test_num = int(lines[0])
    test = [line.strip() for line in lines[1:]]
logging.info(f"Test assignments: {test}")

def updateTxtFile():
    with open(train_txt_path, 'w') as f:
        f.write(f"{train_num}\n")
        for item in train:
            f.write(f"{item}\n")
    with open(test_txt_path, 'w') as f:
        f.write(f"{test_num}\n")
        for item in test:
            f.write(f"{item}\n")

# Function to download a folder and process images
def download_folder_and_analyze(folder_id, save_path):
    request = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id,name,mimeType)").execute()
    files = request.get('files', [])
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_mime_type = file.get('mimeType', '')
        next_path = Path(f"{save_path}/{file_name}")

        if file_mime_type == 'application/vnd.google-apps.folder':
            os.makedirs(next_path, exist_ok=True)
            download_folder_and_analyze(file_id, next_path)
        else:
            download_file(file_id, next_path)

# Function to download a specific file
def download_file(file_id, save_path):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        if not save_path.exists():
            with open(save_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logging.info(f"Downloading {file_id}: {int(status.progress() * 100)}%")
        else:
            logging.info(f"{save_path} exists, download skipped")
    except HttpError as error:
        logging.error(f"An error occurred: {error} \nFile {file_id} failed")
        return None

# Download the submissions folder and process
request = drive_service.files().list(q=f"'{SUBMISSION_FOLDER_ID}' in parents", fields="files(id,name)").execute()
files = request.get('files', [])

for file in files:
    file_id = file['id']
    file_name = file['name']
    if file_name in train:
        file_path = Path(os.path.join(DATA_PATH, 'train', file_name))
        if file_path.is_dir() and SKIP_EXIST_DIR:
            logging.info(f"{file_path} directory already exists... skipping download")
            continue
    elif file_name in test:
        file_path = Path(os.path.join(DATA_PATH, 'test', file_name))
        if file_path.is_dir() and SKIP_EXIST_DIR:
            logging.info(f"{file_path} directory already exists... skipping download")
            continue
    elif train_num // TRAIN_RATIO >= test_num:
        file_path = Path(os.path.join(DATA_PATH, 'test', file_name))
        os.makedirs(file_path, exist_ok=True)
        test_num += 1
        test.append(file_name)
    else:
        file_path = Path(os.path.join(DATA_PATH, 'train', file_name))
        os.makedirs(file_path, exist_ok=True)
        train_num += 1
        train.append(file_name)
    download_folder_and_analyze(file_id, file_path)

updateTxtFile()
