import kivy
import kivy.properties as kyProps
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.config import Config
import time
from kivy.core.window import Window
from kivy.properties import NumericProperty,StringProperty
from kivy.uix.widget import Widget
from tkinter import filedialog

from model_api import load_model,predict_model_test,stop_model,predict_model
from temp_model import NeuralNet

# ai model object
model = None

# loading kv language file
Builder.load_file('TechTutor.kv')

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
        predict_model_test(model,"src/mbrimberry_files")
        
    # method for when stop button is pressed
    def stop_press(self):
        stop_model()

    # method for when change grade key button is pressed
    def change_key_button(self):
        key_file = self.ids.grade_key_image.source
        # using tkinter to create a field dialog so it looks like default system file explorer
        new_key_file = filedialog.askopenfile(initialdir="/", 
            title = "Select a new grade key",
            filetypes=(("PNG","*.png"),
                        ("JPG","*.jpg"),
                        ("JPEG","*.jpeg"),
                        ("HEIC","*.heic"),
                        ("All Files","*.*")))       
        if(new_key_file is None):
            pass
        elif not (new_key_file.name.endswith(('.png','.jpg','.jpeg','.heic'))):
            self.error_file = 'TechTutor only accepts .png .jpeg .jpg or .heic files'
        else:
            self.error_file = ''
            key_file = new_key_file.name
            self.ids.grade_key_image.source = key_file
    
    # method for when pause button is pressed    
    def pause_press(self):
        print("here")
        
    # Change Account Info
    #====================================================================================
    # Opens a file explorer to select a json file    
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
            self.save_config(json_file.name)

    # Saves the path of the selected json from the def above to a specified file
    def save_config(self, file_path):
        # Define the configuration file path
        config_file_path = 'src\json_config.txt'

        # Write the selected path to a configuration file
        with open(config_file_path, 'w') as config_file:
            config_file.write(file_path)

        # Update the UI or console to confirm the file was saved
        print(f"Configuration file path saved: {file_path}")
        self.load_config()
    
    # Reads from the file and returns the string    
    def load_config(self):
        print("Loading string from config")
        try:
            with open('src\json_config.txt', 'r') as config_file:
                file_path = config_file.read().strip()
                print(f"Loaded file path from config: {file_path}")
                return file_path
        except FileNotFoundError:
            print("Configuration file not found.")
            return None
    #====================================================================================       
    

# main call loop for kivy to make application window
class TechTutorApp(App):
    set_value = 5
    def build(self):
        
        # setting window background to white
        Window.clearcolor = (49/255,51/255,56/255,1)
        return MyFloatLayout()
    global model 
    model = load_model("./model_test1.pt")
    
if __name__ == '__main__':
    TechTutorApp().run()