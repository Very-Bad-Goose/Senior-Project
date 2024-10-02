"""
    Testing page for the google_sheet.py file
    Tests to make sure that we can retrieve and set cell values, row values, and column values
    
    For now it's pretty basic but can add more testability later
"""


from google_sheet import google_sheet
import os

def test_google_sheet_operations(credentials_json, sheet_id):
    sheet = google_sheet(credentials_json, sheet_id)
    
    # Test reading from the sheet
    print("Testing read operations...")
    try:
        # Get a value from a specific cell (e.g., row 2, col 3)
        cell_value = sheet.get_cell(2, 3)
        print(f"Value from cell (2, 3): {cell_value}")
        
        # Get an entire row (e.g., row 2)
        row_values = sheet.get_row(2)
        print(f"Values from row 2: {row_values}")
        
        # Get an entire column (e.g., column 1)
        col_values = sheet.get_col(1)
        print(f"Values from column 1: {col_values}")


    except Exception as e:
        print(f"Read operation failed: {e}")
    
    # Test write operations
    print("\nTesting write operations...")
    try:
        # Update a cell with some test value
        sheet.update_cell(3, 1, 'Test Update')
        print("Cell updated successfully.")
        
        # Add a new row with test data
        sheet.add_row(['New', 'Test', 'Row'])
        print("New row added successfully.")
        
    except Exception as e:
        print(f"Write operation failed: {e}")
    
    

if __name__ == '__main__':
    # Path to your Google service account credentials JSON file
    credentials_json = 'path/to/credentials.json'  
    
    # The Google Sheet ID you want to work with (can be found in the sheet's URL)
    sheet_id = 'your_google_sheet_id_here'
    
    # Run the test
    test_google_sheet_operations(credentials_json, sheet_id)