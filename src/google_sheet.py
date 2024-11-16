# Collin Dunkle
#
# This is a class that we can use to retrieve information from a google sheet. For now we only
# have a few functions but those can be increased from ease in here
# NOTE: Uses the gspread lib, you can install it with 'pip install gspread'

# Possible additions to add in here:
# Logging functionality
# Execution time
# Processed/Unprocessed
# Feedback is already taken care of

# Additional functionality
# Google Sheets API has a requests per minute limit, for users it's only 100 requests per
# minute, and service accounts it's 500.
# Need a function that if a requests max is reached then it starts a timer and waits, 
# once the timer is complete it starts the process again


import gspread
import re
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
from gspread.exceptions import APIError
from googleapiclient.discovery import build
import time
from datetime import datetime
from logger import SheetLogger



class google_sheet:
    def __init__(self, credentials_json, sheet_id):
        self.credentials_json = credentials_json
        self.sheet_id = sheet_id
        self.client = self._authenticate()
        print(f"Authenticated client: {self.client}")
        self.sheet = self._initialize_sheet()
        self.worksheet = self.get_worksheet()
        self.logsheet = self.check_or_create_log_sheet("Log")
        self.logger = SheetLogger(self, self.logsheet)
    


    # Uncomment above to add text api key readability        
    """ 
    api_key_file_path = '' # TODO add file path for your api.txt file
    
    def get_api_key_from_file(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                return lines[0].strip()
        except Exception as e:
            print(f"Failed to read api key from file: {e}")
            return None
        
    API_KEY = get_api_key_from_file(api_key_file_path)
    """ 
    # Authenticates with the google sheet object using json credentials and api key
    def _authenticate(self):
        SCOPES = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_json, scopes=SCOPES)
        self.drive_service = build('drive', 'v3', credentials=credentials)  # Add Drive service
        service = build('sheets', 'v4', credentials=credentials)
        return gspread.authorize(credentials)
    
    # Inits the sheet for us to use later
    def _initialize_sheet(self):
        return self.client.open_by_key(self.sheet_id)
    
    # Grabs the worksheet (default 0) for us
    def get_worksheet(self, index=0):
        return self.sheet.get_worksheet(index)
    
    # Return a specified cell
    def get_cell(self, row, col):
        return self.worksheet.cell(row,col).value
    
    # Return a specified row
    def get_row(self, row):
        return self.worksheet.row_values(row)
    
    # Return a specified column
    def get_col(self, col):
        return self.worksheet.col_values(col)
    
    #Returns true or false if value is the same as the cell's value
    def compare_cell(self, value, row, col):
        return value == self.get_cell(row, col)

    # Check to see if we have a log sheet, return if it exists, if not then create one
    def check_or_create_log_sheet(self, log_sheet_name="Log"):
        try:
            # Check if the log sheet exists by trying to access it
            try:
                log_sheet = self.sheet.worksheet(log_sheet_name)
                print(f"Log sheet '{log_sheet_name}' exists.")
                return log_sheet
            except gspread.WorksheetNotFound:
                # If the log sheet doesn't exist, create it
                log_sheet = self.sheet.add_worksheet(title=log_sheet_name, rows=1000, cols=10)
                # Create top of log sheet
                # TODO
                print(f"Log sheet '{log_sheet_name}' created successfully.")
                return log_sheet
        except Exception as e:
            return None
    

    """"
    DEPRECATED FUNCTION

    # Returns an array of rows that have not been marked as processed by the program iterating through it
    def get_unprocessed_rows(self, start_row, end_row, status_col):
        operation_name = "Get Unprocessed Rows"
        try:
            unprocessed_rows = []
            for row in range(start_row, end_row + 1):
                status = self.worksheet.cell(row, status_col).value
                # TODO Change this based on column for "Processed flag"
                if status != 1:      
                    unprocessed_rows.append(row)
            print(f"Unprocessed rows: {unprocessed_rows}")
            return unprocessed_rows
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error fetching unprocessed rows: {e}")
            return []
    """
    
    # Write operations
    
    # Updates a singular cell
    def update_cell(self, row, col, string):
        operation_name = "Update Cell"
        try:
            result = self.logger.time_operation(self.retry_on_rate_limit(self.worksheet.update_cell, row, col, string))
            self.logger.log(operation_name, result)
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error writing to cell: {e}")
    
    """
    DEPRECATED FUNCTION
    
    # Marks the cell that keeps track if a row has been processed or not as processed
    def mark_processed(self, row, col):
        operation_name = "Mark Processed"
        try:
            message_value = 1   # TODO change to value of column expectation
            self.update_cell(row,col,message_value)
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error marking cell as processed: {str(e)}")
    """
    # Logs the result of an operation to the log worksheet
    def log_result(self, message):
        try:
            log_sheet = self.logsheet
            log_sheet.append_row(message)
            print(f"Logged message: {message}")
        except Exception as e:
            print(f"Error logging message: {e}")
            
    # Timer function that will cause the process to wait for 60 seconds        
    def wait_fifteen(self):
        print("Waiting for 15 seconds...")
        for remaining in range(15, 0, -1):
            print(f"Time remaining: {remaining} seconds", end='\r')
            time.sleep(1)
        print("\nFifteen Seconds have passed.")

    # Wrapper to go around our update cell call so that we can wait and not lose progress
    def retry_on_rate_limit(self, func, *args, **kwargs):
        while True:
            try:
                print("Running function")
                return func(*args, **kwargs)
            except APIError as error:
                print("Hit Quota Max")
                if error.response.status_code == 429:
                    print("Retrying after wait!")
                    self.wait_fifteen()
                else:
                    raise

    # Google Drive Operations
    @staticmethod
    def extract_folder_id(drive_url):
        match = re.search(r"drive\.google\.com\/drive\/folders\/([a-zA-Z0-9_-]+)", drive_url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid Google Drive folder URL")
        
    def list_drive_folder_contents(self, folder_url):
        folder_id = self.extract_folder_id(folder_url)
        try:
            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name)"
            ).execute()
            files = results.get('files', [])
            return [{"name": file['name'], "id": file['id']} for file in files]
        except HttpError as error:
            print(f"Error accessing Drive folder: {error}")
            return []
    
    # Searches the Root Directory (Parent) for a subfolder by name
    # Based on the client's policies, the folder names should always be:
    # "Desk Images" and "Activity Packet"
    # Inside the "Desk Images" Folder there should be "desk_1.png"
    def get_folder_id_by_name(self, parent_folder_id, folder_name):
        """Get the ID of a subfolder by its name."""
        try:
            query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']  # Assuming folder names are unique
            else:
                raise ValueError(f"Folder '{folder_name}' not found.")
        except HttpError as error:
            print(f"Error fetching folder '{folder_name}': {error}")
            return None

    def list_png_files_in_folder(self, folder_id):
        """List all PNG files in a given folder."""
        try:
            query = f"'{folder_id}' in parents and mimeType='image/png'"
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name, webContentLink)"
            ).execute()
            files = results.get('files', [])
            return [{"name": file['name'], "id": file['id'], "url": file.get('webContentLink')} for file in files]
        except HttpError as error:
            print(f"Error fetching PNG files: {error}")
            return []