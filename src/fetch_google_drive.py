from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import os
from pathlib import Path
SKIP_EXIST_DIR = 1
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = '.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#This is the ID of a test drive I made
#TODO Change to clients drive
SUBMISSION_FOLDER_ID = "1-SFFGspWjvHu0JKsqoWUinpdZtRAqs1v"
# path to where the test and training data will be created if it does not exists
DATA_PATH = '\submissions'
# How many training images for each test image
TRAIN_RATIO = 4
test_path = DATA_PATH + '/test'
train_path = DATA_PATH + '/train'
#IMPORTANT make sure to have the updated txt files named train.txt and test.txt from 
train_txt_path = DATA_PATH + "/train/train.txt"
test_txt_path = DATA_PATH + "/test/test.txt"
os.makedirs(train_path, exist_ok=True)  
os.makedirs(test_path, exist_ok=True) 
drive_service = build('drive', 'v3', credentials=credentials)
current_dir = DATA_PATH
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    train_num = int(lines[0])
    #List of assignments (Labeled by dates) that should go into trainning folder
    train  = [line.strip() for line in lines[1:]]
print(train)

with open(test_txt_path, 'r') as f:
    lines = f.readlines()
    test_num = int(lines[0])
    #List of assignments (Labeled by dates) that should go into trainning folder
    test  = [line.strip() for line in lines[1:]]
print(test)

def updateTxtFile():
    with open(train_txt_path, 'w') as f:
        f.write("%d\n" % train_num)
        for item in train:
            f.write("%s\n" % item)
    with open(test_txt_path, 'w') as f:
        f.write("%d\n" % test_num)
        for item in test:
            f.write("%s\n" % item)

# Function to download a folder and process images
def download_folder_and_analyze(folder_id, save_path):
    request = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id,name,mimeType)").execute()
    files = request.get('files', [])
    current_dir = save_path
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_mime_type = file.get('mimeType', '')
        next = Path(f"{current_dir}/{file_name}")
        if file_mime_type == 'application/vnd.google-apps.folder':
            
            os.makedirs(next, exist_ok=True)  
            download_folder(file_id, next)
        else:
            download_file(file_id, next)

# Function to download a specific file
def download_file(file_id, save_path):
    try:
        request = drive_service.files().get_media(fileId=file_id)      
        if not save_path.exists():
            with open(save_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Downloading {file_id}: {int(status.progress() * 100)}%")
        else:
            print(f"{save_path} exists, download skipped")
    except HttpError as error:
        print(f"An error occurred: {error} \nFile {file_id} failed")
        return None

# Download the submissions folder and process
request = drive_service.files().list(q=f"'{SUBMISSION_FOLDER_ID}' in parents", fields="files(id,name)").execute()
files = request.get('files', [])

for file in files:
    file_id = file['id']
    file_name = file['name']
    if file_name in train:
        file_path = Path(f"{DATA_PATH}/train/{file_name}")
        if file_path.is_dir() and SKIP_EXIST_DIR:
            print(f"{file_path} directory already exists... skipping download")
            continue
    elif file_name in test:
        file_path = Path(f"{DATA_PATH}/test/{file_name}")
        if file_path.is_dir() and SKIP_EXIST_DIR:
            print(f"{file_path} directory already exists... skipping download")
            continue
    elif train_num // TRAIN_RATIO >= test_num:
        file_path = Path(f"{DATA_PATH}/test/{file_name}")
        os.makedirs(file_path, exist_ok=True)  
        test_num = test_num + 1
        test.append(file_name)
    else:
        file_path = Path(f"{DATA_PATH}/train/{file_name}")
        os.makedirs(file_path, exist_ok=True)  
        train_num = train_num + 1   
        train.append(file_name)
    download_folder(file_id,file_path)
updateTxtFile()
