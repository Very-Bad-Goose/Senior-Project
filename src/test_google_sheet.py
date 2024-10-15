"""
    Testing page for the google_sheet.py file
    Tests to make sure that we can retrieve and set cell values, row values, and column values
    
    For now it's pretty basic but can add more testability later
    
    Test template
    # Test Case 1:
    # Do operation
    # Success case -
    # Result -
    # Date - 
    
    # Warnings:
        Update_cell is marked as a failure in logs, but still writes to sheet
    
    


"""

import json
from google_sheet import google_sheet
import os

def test_google_sheet_operations(credentials_json, sheet_id):
    sheet = google_sheet(credentials_json, sheet_id)
    
    # Test reading from the sheet
    print("Testing read operations...")
    try:
        # Test Case 1:
        # Read from single cell
        # Success case - Correctly returns string in specified cell
        # Result - Success
        # Date - 10/15/2024
        cell_value = sheet.get_cell(2, 3)
        print(f"Value from cell (2, 3): {cell_value}")
        
        # Test Case 2:
        # Read from entire row
        # Success case - Correctly returns array of strings for entire row
        # Result - Success
        # Date - 10/15/2024
        row_values = sheet.get_row(2)
        print(f"Values from row 2: {row_values}")
        
        # Test Case 3:
        # Read from entire column
        # Success case - Correctly returns values in specified column
        # Result - Success
        # Date - 10/15/2024
        col_values = sheet.get_col(1)
        print(f"Values from column 1: {col_values}")


    except Exception as e:
        print(f"Read operation failed: {e}")
    
    # Test write operations
    print("\nTesting write operations...")
    try:
        # Test Case 4:
        # Write to single cell
        # Success case - Correctly writes to specified cell
        # Result - Success
        # Date - 10/13/2024
        sheet.update_cell(3, 1, 'Test Update')
        print("Cell updated successfully.")
        
    except Exception as e:
        print(f"Write operation failed: {e}")

# Function to test on sheet with current data format
def test_populated_sheet(credentials_json, sheet_id):
    sheet = google_sheet(credentials_json, sheet_id)
    """
    # Test Case 1:
    # Read from cell (1,1)
    # Success case - Successfully print cell value
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 1:")
    test1 = sheet.get_cell(1,1)
    print(f"Value from cell 1,1: {test1}")
    
    
    # Test Case 2:
    # Read entire column of A
    # Success case - Returns correct values for column A
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 2:")
    test2 = sheet.get_col(1)
    print(test2)
    
    
    # Test Case 3:
    # Read entire row of 3
    # Success case - Successfully reads row 3 to console
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 3:")
    test3 = sheet.get_row(3)
    print(test3)
    
    
    # Test Case 4:
    # Compares string from cell to 'False'
    # Success case - Returns false for any cell that does not have 'False'
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 4:")
    test4 = sheet.compare_cell("False",1,1)
    print(test4)
    
    
    # Test Case 5:
    # Mark Processed Cell
    # Success case - Marks a row as processed
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 5:")
    test5 = sheet.mark_processed(1,1)
    print(test5)
    
    
    # Test Case 6:
    # Loop through a row and update cells
    # Success case - Updates a col successfully
    # Result - Failure
    # Date - 10/15/2024
    print("Testing Case 6:")
    for row in range(1, 10):
        sheet.update_cell(row,5, "Test Case 6")
    test6 = sheet.get_col(5)
    print(test6)      
    
    
    # Test Case 7:
    # Loop through a col and update cells 
    # Success case - Updates a row successfully
    # Result - Success 
    # Date - 10/15/2024
    print("Testing Case 7:")
    for col in range(1,10):
        sheet.update_cell(5, col, "Test Case 7")
    test7 = sheet.get_row(5)
    print(test7)
    """
    
    # Test Case 8:
    # Loop through a 2d array and update cells, then mark as processed
    # Success case - Updates 2d array correctly and marks them as processed
    # Result - Write limit exceeded for Google API, will try again after time limit
    # Date - 10/15/2024
    
    print("Testing Case 8:")
    for col in range(11,20):
        for row in range(11,20):
            sheet.update_cell(row,col, "Test Case 8")
            #sheet.mark_processed(row, col)
    print("Test 8 complete")
    
    """
    # Test Case 9:
    # Get Unprocessed Cells
    # Success case - Successfully returns list of unprocessed cells
    # Result - 
    # Date - 
    print("Testing Case 9:")
    #test9 = sheet.get_unprocessed_rows()
    
    # Test Case 10:
    # Compare Cell True Case
    # Success case - Successfully returns true for cell comparison
    # Result - Success
    # Date - 10/15/2024
    print("Testing Case 10:")
    test10 = sheet.compare_cell("Test Case 6", 1, 5)
    print(test10)
    """

    
    




# Main Call
if __name__ == '__main__':
    # Path to your Google service account credentials JSON file
    credentials_json = 'E:\senior-project-435718-4245b09a4eb5.json'
    
    # The Google Sheet ID you want to work with (can be found in the sheet's URL)
    sheet_id = "1HqZEuFXYvQLKor-9IaBj3J9--2XCa7MIA8GaTNpxkUg"
    #Google sheet with data
    sheet_id_2 = ""
    
    # Run the test
    #test_google_sheet_operations(credentials_json, sheet_id)
    test_populated_sheet(credentials_json, sheet_id)

