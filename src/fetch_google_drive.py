#Joshua Grindstaff
#IMPORTANT need to add service account email to drive, think we'll need to contact client for that
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import os
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
#TODO Replace with CreditialFile (Same as test2)
SERVICE_ACCOUNT_FILE = 'apiketget-ad9721240cbb.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#This is the ID of a test drive I made
#TODO Test this ID
SUBMISSION_FOLDER_ID = "1o0w7mOrhZh5tCuaVXnpBa8195J5iUBz2"
# path to where the test and training data will be created if it does not exists
DATA_PATH = 'submissions'
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

with open(test_txt_path, 'r') as f:
    lines = f.readlines()
    test_num = int(lines[0])
    #List of assignments (Labeled by dates) that should go into trainning folder
    test  = [line.strip() for line in lines[1:]]

def updateTxtFile():
    with open(train_txt_path, 'w') as f:
        f.write("%d\n" % train_num)
        for item in train:
            f.write("%s\n" % item)
    with open(test_txt_path, 'w') as f:
        f.write("%d\n" % test_num)
        for item in test:
            f.write("%s\n" % item)

#Function to download all contents in a specified folder
def download_folder(folder_id, save_path):
    request = drive_service.files().list(q=f"'{folder_id}' in parents",fields="files(id,name,mimeType)").execute()
    files = request.get('files', [])
    current_dir = save_path
    for file in files:
        #grab file id and name
        file_id = file['id']
        file_name = file['name']
        file_mime_type = file.get('mimeType', '')
        if file_mime_type == 'application/vnd.google-apps.folder':
            os.makedirs(current_dir + "/" + file_name, exist_ok=True)  
            download_folder(file_id, current_dir + "/" + file_name)
        else:
            download_file(file_id, current_dir + "/" + file_name)

    
#Function to download a specific file to a specific local directory
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
    
# Downloading the submissions folder
request = drive_service.files().list(q=f"'{SUBMISSION_FOLDER_ID}' in parents",fields="files(id,name)").execute()
files = request.get('files', [])
for file in files:
    #grab file id and name
    file_id = file['id']
    file_name = file['name']
    if file_name in train:
        download_folder(file_id,DATA_PATH + "/train/" + file_name)
    elif file_name in test:
        download_folder(file_id,DATA_PATH + "/test/"+ file_name)
    elif train_num // TRAIN_RATIO >= test_num:
        os.makedirs(DATA_PATH + "/test/"+ file_name, exist_ok=True)  
        download_folder(file_id,DATA_PATH + "/test/"+ file_name)
        test_num = test_num + 1
        test.append(file_name)
    else:
        os.makedirs(DATA_PATH + "/train/"+ file_name, exist_ok=True)  
        download_folder(file_id,DATA_PATH + "/train/"+ file_name)
        train_num = train_num + 1   
        train.append(file_name)
updateTxtFile()

