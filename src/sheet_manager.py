from google_sheet import google_sheet

class SheetController:
    def __init__(self, credentials_json, sheet_id, user_data, data_processor):
        """
        Initializes the controller with a Google Sheet instance and other supporting classes.
        Args:
            credentials_json (str): Path to Google service account credentials JSON file.
            sheet_id (str): Google Sheet ID.
            user_data (UserData): An instance of a class holding user data.
            data_processor (DataProcessor): An instance of a class for data processing.
        """
        self.sheet = google_sheet(credentials_json, sheet_id)
        self.user_data = user_data
        self.data_processor = data_processor

    def retrieve_user_data(self, user_id):
        """Retrieve data for a specific user and update the sheet with processed information."""
        data = self.user_data.get_user_info(user_id)
        if data:
            # Process data before sending it to the sheet
            processed_data = self.data_processor.process(data)
            row = self.find_row_for_user(user_id)
            if row:
                for col, value in enumerate(processed_data, start=1):
                    self.sheet.update_cell(row, col, value)
                print(f"Updated user {user_id} data in row {row}.")
            else:
                print(f"User {user_id} not found.")
        else:
            print(f"No data found for user {user_id}.")

    def find_row_for_user(self, user_id):
        """Find the row in the sheet where the user ID is stored (for example, in the first column)."""
        col_data = self.sheet.get_col(1)  # Assuming column 1 holds user IDs
        for idx, cell_value in enumerate(col_data, start=1):
            if cell_value == user_id:
                return idx
        return None

    def update_user_status(self, user_id, status):
        """Update the status of a user in the sheet."""
        row = self.find_row_for_user(user_id)
        if row:
            self.sheet.update_cell(row, 2, status)  # Assuming column 2 is for status
            print(f"Updated status for user {user_id} to '{status}'.")
        else:
            print(f"User {user_id} not found.")

    def sync_all_users(self):
        """Retrieve all user data and update the sheet."""
        all_users = self.user_data.get_all_users()
        for user_id, user_info in all_users.items():
            processed_data = self.data_processor.process(user_info)
            row = self.find_row_for_user(user_id)
            if row:
                for col, value in enumerate(processed_data, start=1):
                    self.sheet.update_cell(row, col, value)
                print(f"Synchronized data for user {user_id} in row {row}.")
            else:
                print(f"User {user_id} not found in the sheet.")

    def get_user_info_from_sheet(self, user_id):
        """Retrieve and process a user's data directly from the sheet."""
        row = self.find_row_for_user(user_id)
        if row:
            row_data = self.sheet.get_row(row)
            return self.data_processor.process(row_data)
        return None

# Example supporting classes:
class UserData:
    def __init__(self, user_data):
        """Initialize with a dictionary of user data where keys are user IDs."""
        self.user_data = user_data

    def get_user_info(self, user_id):
        """Retrieve specific user data."""
        return self.user_data.get(user_id)

    def get_all_users(self):
        """Retrieve all user data."""
        return self.user_data

class DataProcessor:
    def process(self, data):
        """Process data (e.g., clean up, format) before using it."""
        # Example processing logic
        return [str(d).strip().capitalize() for d in data] if isinstance(data, list) else str(data).strip()

# Example usage:
if __name__ == '__main__':
    credentials_json = 'E:\\senior-project-435718-4245b09a4eb5.json'
    sheet_id = "1HqZEuFXYvQLKor-9IaBj3J9--2XCa7MIA8GaTNpxkUg"
    
    # Example user data
    user_data = UserData({
        'user1': ['Alice', 'Active'],
        'user2': ['Bob', 'Inactive']
    })

    # Data processor for formatting
    data_processor = DataProcessor()

    # Initialize controller
    controller = SheetController(credentials_json, sheet_id, user_data, data_processor)

    # Perform operations
    controller.retrieve_user_data('user1')
    controller.update_user_status('user2', 'Active')
    controller.sync_all_users()
    user_info = controller.get_user_info_from_sheet('user1')
    print(f"Retrieved info for user1 from sheet: {user_info}")
