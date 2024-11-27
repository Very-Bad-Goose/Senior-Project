import kivy
import kivy.properties as kyProps
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.config import Config
import time
import os
import shutil
from kivy.core.window import Window
from kivy.properties import NumericProperty,StringProperty
from kivy.uix.widget import Widget
from tkinter import filedialog
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from kivy.clock import Clock
from model_api import load_model,predict_model_test,stop_model,predict_model

# ai model object
model:FasterRCNN

# loading kv language file
Builder.load_file('./src/TechTutor.kv')

# disbaling touch screen emulation on mouse
Config.set("input","mouse","mouse,disable_multitouch")

class MyProgressBar(Widget):
    set_value = NumericProperty(0)

# custom layout to hold all UI elements using a KV file for UI elements
class MyFloatLayout(FloatLayout):
    progress_bar_value = NumericProperty(0)
    error_file = StringProperty("")
    # method for when start button is pressed
    def start_press(self):
        self.ids.progress_bar_background.set_value += 10
        if(self.ids.progress_bar_background.set_value > 100):
            self.ids.progress_bar_background.set_value = 0
        self.progress_bar_value = self.ids.progress_bar_background.set_value
        
        global model
        predict_model(model,"./src/mbrimberry_files/")
        
    # method for when stop button is pressed
    def stop_press(self):
        stop_model()
        
    # Change Account Info
    #====================================================================================
    # Opens a file explorer to select a json file    
    # Saves the file to src/config_files and then saves the path to the config file as well
    def select_json_file(self):
        # Open file dialog for JSON file selection
        json_file = filedialog.askopenfile(
            initialdir="/", 
            title="Select configuration file",
            filetypes=(("JSON files", "*.json"), ("All Files", "*.*"))
        )
        
        if json_file is None:
            return  # User cancelled the file selection

        elif not json_file.name.endswith('.json'):
            self.error_file = 'Only .json files are accepted for configuration'
        else:
            self.error_file = ''
            # Save the selected JSON file path to a configuration file
            self.save_json_file(json_file.name)
            
    # Save the JSON file to a designated folder and return its new path
    def save_json_file(self, file_path):
        config_dir = 'src/config_files'
        os.makedirs(config_dir, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(config_dir, file_name)
        
        try:
            shutil.copy(file_path, destination_path)
            print(f"JSON file saved to: {destination_path}")
            self.save_config(destination_path)
        except Exception as e:
            print(f"Failed to save JSON file: {e}")
            return None

    # Save the path of the saved JSON file to json_config.txt
    def save_config(self, file_path):
        config_file_path = 'src/json_config.txt'

        try:
            with open(config_file_path, 'w') as config_file:
                config_file.write(file_path)
            print(f"Configuration file path saved to {config_file_path}: {file_path}")
            self.load_config()
        except Exception as e:
            print(f"Failed to save configuration file: {e}")

    # Read and return the saved configuration path from json_config.txt
    def load_config(self):
        config_file_path = 'src/json_config.txt'
        try:
            with open(config_file_path, 'r') as config_file:
                file_path = config_file.read().strip()
                print(f"Loaded file path from config: {file_path}")
                return file_path
        except FileNotFoundError:
            print("Configuration file not found.")
            return None

    def save_sheet_id(self):
            sheet_id = self.ids.sheet_id_input.text
            temp = sheet_id.split("/")
            print(temp)
            sheet_id = temp[5]
            print(sheet_id)
            file_path = './sheet_id.txt'
    
            try:
                # Open the file in append mode to add new IDs on new lines
                with open(file_path, 'w') as file:
                    file.write(sheet_id)  # Add a newline after each ID
                    print(sheet_id)
                self.error_file = "Sheet ID saved successfully!"
    
                # Clear the message after 2 seconds
                Clock.schedule_once(self.clear_message, 2)
            except Exception as e:
                self.error_file = f"Error saving Sheet ID: {e}"
    
    def clear_message(self, *args):
        self.error_file = ""  # Clears the message

    #====================================================================================       
    

# main call loop for kivy to make application window
class TechTutorApp(App):
    set_value = 5
    def build(self):
        
        # setting window background to white
        Window.clearcolor = (49/255,51/255,56/255,1)
        return MyFloatLayout()
    global model 
    model = load_model("./models/id_periodNum_model.pt")
    
if __name__ == '__main__':
    TechTutorApp().run()
