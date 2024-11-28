import os
from google_sheet import google_sheet

class SheetController:
    def __init__(self, credentials_config_path, sheet_id_config_path):
        """
        Initializes the controller with a Google Sheet instance and other supporting classes.
        Args:
            credentials_config_path (str): Path to the configuration file for credentials JSON.
            sheet_id_config_path (str): Path to the configuration file for the Google Sheet ID.
            
        """
        # Read configurations from files
        self.credentials_json = self.read_config(credentials_config_path)
        self.sheet_id = self.read_config(sheet_id_config_path)
        
        # Initialize the Google Sheet instance
        self.sheet = google_sheet(self.credentials_json, self.sheet_id)

    @staticmethod
    def read_config(file_path):
        """
        Reads a single line from a configuration file.
        Args:
            file_path (str): Path to the configuration file.
        Returns:
            str: The content of the file as a string.
        """
        try:
            with open(file_path, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Configuration file {file_path} not found.")
            return None
        
    def ExampleFunction1(self, cell_value):
            # Example placeholder function for criteria check
            # This function can be customized to perform various checks on the cell value
            from object_detection_model import predict_with_model
            from main import get_caddy_model,get_desk_model,get_packet_model
            
            print("running")
            from handwriting_recognition import process_image_to_digits
            print("also running")
            
            # model_path = "./models/id_periodNum_model.pt"
            # print(model_path)
            
            # Change this to get the file path from the cell
            img = './src/mbrimberry_files/Submissions/03 12 2024/Activity  574644 - 03 12 2024/Activity Packet/activity_1.png'
            print(img)
            bbox = predict_with_model(img,get_packet_model())
            id = process_image_to_digits(img,bbox[0])
            print(id)
            return cell_value.lower() == "example"  # Modify as needed

    def process_sheet(self):
        """
        Loops through the Google Sheet rows and performs comparisons.
        """
        
        rows = self.sheet.get_all_rows()  # Assuming google_sheet has a method to fetch all rows
        for row in rows:
            if len(row) < 4:
                continue  # Skip rows with fewer than 4 columns
            col1_value = row[0]  # Column 1 value
            col4_value = row[3]  # Column 4 value
            result = self.ExampleFunction1(col4_value)
            print(f"Row {rows.index(row) + 1}: Column 4 value: '{col4_value}', "
                f"Column 1 value: '{col1_value}', Comparison result: {result}")
            # Perform action based on comparison result (e.g., log, flag, or update the sheet)