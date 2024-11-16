# Class that tests the current google_sheet class functionality by running automated unit tests
# on it. Mainly for testing access with using the Google Drive API

import unittest
from unittest.mock import MagicMock, patch
import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from googleapiclient.errors import HttpError
from google_sheet import google_sheet  # Replace with your actual module name

class TestGoogleSheet(unittest.TestCase):
    @patch("google_sheet.service_account.Credentials.from_service_account_file")
    @patch("google_sheet.build")
    @patch("google_sheet.gspread.authorize")
    def setUp(self, mock_authorize, mock_build, mock_credentials):
        # Mock Google API credentials
        mock_credentials.return_value = MagicMock()
        mock_authorize.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        # Initialize the class with mock data
        self.mock_credentials_json = "mock_credentials.json"
        self.mock_sheet_id = "mock_sheet_id"
        self.google_sheet = google_sheet(self.mock_credentials_json, self.mock_sheet_id)

        # Mocking sheet and worksheet
        self.mock_sheet = MagicMock()
        self.mock_worksheet = MagicMock()
        self.google_sheet.sheet = self.mock_sheet
        self.google_sheet.worksheet = self.mock_worksheet

        # Mocking logger
        self.google_sheet.logger = MagicMock()

    def test_get_cell(self):
        self.mock_worksheet.cell.return_value.value = "Test Value"
        result = self.google_sheet.get_cell(1, 1)
        self.mock_worksheet.cell.assert_called_once_with(1, 1)
        self.assertEqual(result, "Test Value")

    def test_get_row(self):
        self.mock_worksheet.row_values.return_value = ["Row1", "Row2"]
        result = self.google_sheet.get_row(1)
        self.mock_worksheet.row_values.assert_called_once_with(1)
        self.assertEqual(result, ["Row1", "Row2"])

    def test_get_col(self):
        self.mock_worksheet.col_values.return_value = ["Col1", "Col2"]
        result = self.google_sheet.get_col(1)
        self.mock_worksheet.col_values.assert_called_once_with(1)
        self.assertEqual(result, ["Col1", "Col2"])

    def test_check_or_create_log_sheet(self):
        self.mock_sheet.worksheet.return_value = MagicMock()
        result = self.google_sheet.check_or_create_log_sheet("Log")
        self.mock_sheet.worksheet.assert_called_once_with("Log")
        self.assertIsNotNone(result)

        # Test when log sheet does not exist
        self.mock_sheet.worksheet.side_effect = WorksheetNotFound
        self.mock_sheet.add_worksheet.return_value = MagicMock()
        result = self.google_sheet.check_or_create_log_sheet("Log")
        self.mock_sheet.add_worksheet.assert_called_once_with(title="Log", rows=1000, cols=10)
        self.assertIsNotNone(result)

    def test_list_drive_folder_contents(self):
        # Mock Google Drive API response
        self.google_sheet.extract_folder_id = MagicMock(return_value="mock_folder_id")
        self.google_sheet.drive_service.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "123", "name": "File1"}, {"id": "456", "name": "File2"}]
        }
        result = self.google_sheet.list_drive_folder_contents("https://drive.google.com/drive/folders/mock_folder_id")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "File1")
        self.assertEqual(result[1]["name"], "File2")

    def test_extract_folder_id(self):
        url = "https://drive.google.com/drive/folders/123abc"
        folder_id = self.google_sheet.extract_folder_id(url)
        self.assertEqual(folder_id, "123abc")

        with self.assertRaises(ValueError):
            self.google_sheet.extract_folder_id("invalid_url")

if __name__ == "__main__":
    unittest.main()
