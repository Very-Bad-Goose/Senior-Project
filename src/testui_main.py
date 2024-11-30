import main
from main import MyFloatLayout
import unittest
import os
import model_api
from unittest.mock import MagicMock
import pytest

class TestMain(unittest.TestCase):
    # Checks for the exitance of the TechTutor App
    def test_appExists(self):
        app = main.TechTutorApp()
        #print(app)
        self.assertIsNotNone(app)

    def test_guiExists(self):
        app = main.TechTutorApp()
        gui = app.build()
        #print(gui)
        self.assertIsNotNone(gui)

    def test_childrenExist(self):
        app = main.TechTutorApp()
        gui = app.build()
        children = gui.children
        self.assertIsNotNone(children)
        self.assertEqual(len(children), 1)
        for i in children:
            self.assertIsNotNone(i)

    def test_assetsExist(self):
        cwd = os.getcwd()
        img1 = os.path.join(cwd, "src/ui_images", "gray1.png")
        img2 = os.path.join(cwd, "src/ui_images", "gray2.png")
        img3 = os.path.join(cwd, "src/ui_images", "gray3.png")
        img4 = os.path.join(cwd, "src/ui_images", "gray4.png")

        self.assertEqual(os.path.isfile(img1), True)
        self.assertEqual(os.path.isfile(img2), True)
        self.assertEqual(os.path.isfile(img3), True)
        self.assertEqual(os.path.isfile(img4), True)

    def test_assetsMissing(self):
        cwd = os.getcwd()
        missing_img = os.path.join(cwd, "src/ui_images", "missing_image.png")
        self.assertFalse(os.path.isfile(missing_img), "Error: Missing asset test failed.")

    def test_buttonsExist(self):
        app = main.TechTutorApp()
        gui = app.build()
        children = gui.children
        grandchildren = children[0].children
        completion_label = grandchildren[7].text
        change_label = grandchildren[10].text
        stop_label = grandchildren[9].text
        start_label = grandchildren[11].text
        save_sheet = grandchildren[3].text
        
        self.assertEqual(save_sheet,"Save Sheet ID")
        self.assertEqual(completion_label, "0% Done")
        self.assertEqual(stop_label, "Stop")
        self.assertEqual(start_label, "Start")
        self.assertEqual(change_label, "Change Account Credentials")

        #print(grandchildren[0].text) #Save Number
        #print(grandchildren[1].text) #blank
        #print(grandchildren[2].text) #image object
        #print(grandchildren[3].text) #Save Sheet ID
        #print(grandchildren[4].text) #blank
        #print(grandchildren[5].text) #blank
        #print(grandchildren[6].text) #blank
        #print(grandchildren[7].text) #0% Done
        #print(grandchildren[8].text) #MyProgressBar object
        #print(grandchildren[9].text) #Stop
        #print(grandchildren[10].text) #Change Account Credentials
        #print(grandchildren[11].text) #Start

    def setUp(self):
        self.layout = MyFloatLayout()

    def test_start_button(self):
        button = self.layout.ids['start_button']
        self.layout.start_press = MagicMock()
        button.trigger_action()
        self.layout.start_press.assert_called_once()

    def test_stop_button(self):
        button = self.layout.ids['stop_button']
        self.layout.stop_press = MagicMock()
        button.trigger_action()
        self.layout.stop_press.assert_called_once()

if __name__ == '__main__':
    unittest.main()

def test_gui():
    unittest.main()