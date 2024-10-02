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
import time
from datetime import datetime
from logger import SheetLogger

class google_sheet:
    # Init properties
    def __init__(self, credentials_json, sheet_id):
        self.credentials_json = credentials_json
        self.sheet_id = sheet_id
        self.client = self._authenticate()
        self.sheet = self._initialize_sheet()
        self.worksheet = self.get_worksheet()
        self.logsheet = self.check_or_create_log_sheet("Log")
        self.logger = SheetLogger(self, self.logsheet)
        
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

    # Check to see if we have a log sheet, return if it exists, if not then create one
    def check_or_create_log_sheet(self, log_sheet_name="Log"):
        operation_name = "Check or create log sheet"
        try:
            # Check if the log sheet exists by trying to access it
            try:
                log_sheet = self.sheet.worksheet(log_sheet_name)
                print(f"Log sheet '{log_sheet_name}' exists.")
                return log_sheet
            except gspread.WorksheetNotFound:
                # If the log sheet doesn't exist, create it
                log_sheet = self.sheet.add_worksheet(title=log_sheet_name, rows=1000, cols=10)
                print(f"Log sheet '{log_sheet_name}' created successfully.")
                return log_sheet
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error checking or creating log sheet: {e}")
            return None
    

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
    
    # Write operations
    
    # Updates a singular cell
    def update_cell(self, row, col, string):
        operation_name = "Update Cell"
        try:
            result = self.logger.time_operation(self.worksheet.update_cell(row, col, string))
            self.logger.log(operation_name, result)
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error writing to cell: {e}")
    
    # Marks the cell that keeps track if a row has been processed or not as processed
    def mark_processed(self, row, col):
        operation_name = "Mark Processed"
        try:
            message_value = 1   # TODO change to value of column expectation
            result = self.logger.time_operation(self.update_cell(row, col, message_value))
            self.logger.log(operation_name, result)
        except Exception as e:
            self.logger.log_failure(operation_name, str(e))
            print(f"Error marking cell as processed: {str(e)}")

    def log_result(self, message):
        try:
            log_sheet = self.logsheet
            log_sheet.append_row(message)
            print(f"Logged message: {message}")
        except Exception as e:
            print(f"Error logging message: {e}")