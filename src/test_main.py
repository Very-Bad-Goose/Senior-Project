import pytest
from kivy.tests.common import GraphicUnitTest
from main import TechTutorApp

class TestTechTutorApp(GraphicUnitTest):
    
    def setUp(self):
        # Initialize the app instance and build it
        super(TestTechTutorApp, self).setUp()
        self.app = TechTutorApp()
        self.layout = self.app.build()  # Ensure layout is assigned here

    def test_stop_press(self):
        # Simulate stop button press (example case, may require additional setup)
        self.layout.stop_press()

    def test_select_json_file_with_invalid_file(self):
        # Example: Test select_json_file functionality with invalid file
        self.layout.select_json_file()  # Mocking is also needed here
        self.assertEqual(self.layout.error_file, 'Only .json files are accepted for configuration')

    def tearDown(self):
        # Stop the app after tests
        self.app.stop()
