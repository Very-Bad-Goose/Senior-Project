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

import gspread
from google.oauth2 import service_account

class google_sheet:
    
    
    # Init properties
    def __init__(self, credentials_json, sheet_id):
        self.credentials_json = credentials_json
        self.sheet_id = sheet_id
        self.client = self._authenticate()
        self.sheet = self._initialize_sheet()
        self.worksheet = self.get_worksheet() 

    # Set the credentials
    def _authenticate(self):
        credentials_json = None
        # Our scopes, are read and write
        scope = ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/spreadsheets"]
        try:
            credentials = service_account.Credentials.from_service_account_file(
            credentials_json, scopes=scope)
            return gspread.authorize(credentials)
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None

    # Inits the sheet for us to use later
    def _initialize_sheet(self):
        return self.client.open_by_key(self.sheet_id)
    
    # Grabs the worksheet (default 0) for us
    def get_worksheet(self, index=0):
        return self.sheet.get_worksheet(index)
    
    # Return a specified cell
    def get_cell(self, row, col):
        return self.worksheet(row,col).value
    
    # Return a specified row
    def get_row(self, row):
        return self.worksheet.row_values(row)
    
    # Return a specified column
    def get_col(self, col):
        return self.worksheet.col_values(col)
    
    #Returns true or false if value is the same as the cell's value
    def compare_cell(self, value, row, col):
        return value == self.get_cell(row, col)



    
    # Write operations
    
    # Updates a singular cell
    def update_cell(self, row, col, string):
        try:
            self.worksheet.update_cell(row, col, string)
        except Exception as e:
            print(f"Error writing to cell: {e}")
    
