# Code for Google Sheets integration
# (Assuming you have the Google API client library installed)

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Set up the Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = Credentials.from_authorized_user_file('path/to/credentials.json', SCOPES)
service = build('sheets', 'v4', credentials=creds)

# Function to update the spreadsheet with cleanliness score
def update_google_sheets(spreadsheet_id, sheet_name, score):
    values = [[score]]
    body = {'values': values}
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id, range=sheet_name,
        valueInputOption='RAW', body=body
    ).execute()

# Usage
spreadsheet_id = 'your_spreadsheet_id'
sheet_name = 'Sheet1'
image_path = 'path/to/desk_image.jpg'

cleanliness_score = predict_cleanliness(image_path, model)
update_google_sheets(spreadsheet_id, sheet_name, cleanliness_score)

