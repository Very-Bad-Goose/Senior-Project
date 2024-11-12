import unittest
from kivy.tests.common import GraphicUnitTest
from main import TechTutorApp

class TestTechTutorApp(GraphicUnitTest):
    
    def setUp(self):
        # Initialize the app instance and build it
        super(TestTechTutorApp, self).setUp()
        self.app = TechTutorApp()
        self.layout = self.app.build()  # Ensure layout is assigned here

    def test_start_press(self):
        # Simulate start button press and check if the progress bar updates
        self.layout.start_press()
        self.assertEqual(self.layout.progress_bar_value, 10)

    def test_stop_press(self):
        # Simulate stop button press (example case, may require additional setup)
        self.layout.stop_press()
        self.assertIn("here", self.output)

    def test_change_key_button_with_valid_file(self):
        # Example: Test change_key_button functionality
        self.layout.change_key_button()  # Mocking of file dialog is needed here
        # Check if key button logic executes as expected with valid file

    def test_select_json_file_with_invalid_file(self):
        # Example: Test select_json_file functionality with invalid file
        self.layout.select_json_file()  # Mocking is also needed here
        self.assertEqual(self.layout.error_file, 'Only .json files are accepted for configuration')

    def tearDown(self):
        # Stop the app after tests
        self.app.stop()

if __name__ == '__main__':
    unittest.main()
