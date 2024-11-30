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
from model_api import load_model,stop_model,model_predict
from google_sheet import google_sheet
from easyOCR_Number_Recognition import isNumberinResults

# ai model objects
desk_model:FasterRCNN
packet_model:FasterRCNN
caddy_model:FasterRCNN

# tuple of ai models
models:tuple

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

    
        
    # method for when stop button is pressed
    def stop_press(self):
        stop_model()
    
    def create_temp_folder():
        # Get the parent directory of the current directory
        # parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        
        # Path to the Temp folder
        temp_folder_path = "./Temp"
        
        # Create the Temp folder if it doesn't exist
        if not os.path.exists(temp_folder_path):
            try:
                os.makedirs(temp_folder_path)
                print(f"Temporary folder created at: {temp_folder_path}")
            except Exception as e:
                print(f"Error creating temporary folder: {e}")
        else:
            print(f"Temporary folder already exists at: {temp_folder_path}")

    # Call this function where necessary in your script, e.g., before starting your processing
    create_temp_folder()
        
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
        
    
    # method for when start button is pressed
    def start_press(self):
        
        self.ids.progress_bar_background.set_value += 10
        if(self.ids.progress_bar_background.set_value > 100):
            self.ids.progress_bar_background.set_value = 0
        self.progress_bar_value = self.ids.progress_bar_background.set_value
        
        # Grab the json path
        try:
            with open("src/json_config.txt", 'r') as config_file:
                file_path = config_file.read().strip()
                print(f"Loaded file path from config: {file_path}")
        except FileNotFoundError:
            print("Configuration file not found.")
                
        # Grab the sheet path
        try:
            with open("./sheet_id.txt", 'r') as config_file:
                sheet_path = config_file.read().strip()
                print(f"Loaded Sheet ID: {file_path}")
        except FileNotFoundError:
            print("Sheet file not found.")
        
        
        googleSheet_object = google_sheet(file_path, sheet_path)
        
        # MAIN LOOP HERE
        # Start with second row:
        
        # Loop through the rows:
        sheet_row_counter = 2                               # Start with row 2
        row_total = googleSheet_object.get_row_count()      # Get the total amount of populated rows
        
        # Quick CheatSheet for us:
        #----------------------------------------------
        # Column 2 = Student ID
        colStudentID = 2
        # Column 3 = Period Num
        colPeriodNum = 3
        # Column 4 = Assessment Score
        colAssessmentScore = 4
        # Column 5 = Citizenship Score
        colCitizenshipScore = 5
        # Column 6 = Folder URL
        colFolderURL = 6
        # Column 9 = Desk Number
        colDeskNum = 9
        # Column 14 = AI checkbox
        colAICheck = 14
        #----------------------------------------------

        studentID = []
        counter = 2
        while(True):
            # Get the information from the cells for the current row
            temp = googleSheet_object.get_cell(counter, colStudentID)
            if(temp == None ):
                print("breaking at counter: ", counter)
                break
            studentID.append(temp)
            counter += 1
        
        # Begin Loop
        tempFolderPath = "./Temp"
        global models
        while(sheet_row_counter < counter):
            # Get the information from the cells for the current row
            studentID = googleSheet_object.get_cell(sheet_row_counter, colStudentID)
            PeriodNum = googleSheet_object.get_cell(sheet_row_counter, colPeriodNum)
            folderURL = googleSheet_object.get_link(sheet_row_counter, colFolderURL)
            

            # Make sure that the folder url is valid and exists
            if(folderURL != None):
                print(f"getting folderID for {sheet_row_counter}")
                try:
                    folderID = googleSheet_object.extract_folder_id(folderURL)
                    print(folderID)
                    googleSheet_object.download_folder_as_normal_folder(folderID, tempFolderPath)
                except Exception as e:
                    print(f"bad {e}")
            
            # Pass path to the ModelAPIs
            print("Making predictions with models")
            results = []
            t1 = model_predict(models,tempFolderPath,results)
            # Wait for return
            t1.join()
            print("results: ", results)
            if (len(results) < 3):
                print("error, results not populated")
            # Debug print statements
            #print(f"The results in main are: {results[0]}")
            #print(type(results[0]))
            #print(type(results[0][0]))
            #print(f"The results in main are: {results[1]}")
            #print(results[1][0][1])
            #print(results[1][1][1])
            #print(f"The results in main are: {results[2]}")
            #print(results[2][0][1])
            
            # If return 0 for Model APIs for Activity, mark
            # Activity score 0 and mark AI
            packetModelOutput = results[0] #initialize packet model output to list of all page outputs
            try:
                badpacket = False
                # If the studentID or Period Number doesn't match or is unreadable then mark a 0
                for predictions in packetModelOutput:
                    if(studentID != predictions[0]):
                        googleSheet_object.update_cell(sheet_row_counter, colAssessmentScore, 0)
                        googleSheet_object.update_cell(sheet_row_counter, colAICheck, True)
                        badpacket = True
                        break
                    if(PeriodNum != predictions[1]):
                        googleSheet_object.update_cell(sheet_row_counter, colAssessmentScore, 0)
                        googleSheet_object.update_cell(sheet_row_counter, colAICheck, True)
                        badpacket = True
                        break
                #Otherwise, if the Student ID and Period Number is readable give the student a point
                if (badpacket == False):
                    googleSheet_object.update_cell(sheet_row_counter, colAssessmentScore, 1)
                    googleSheet_object.update_cell(sheet_row_counter, colAICheck, True)
           
                # If return 0/null for Model APIs for Desk/Caddy, mark
                # Citizenship Score 0 and mark AI if not already
                try:
                    deskModelOutput = results[1][0][1]
                    if ( deskModelOutput == results[1][1][1]):
                        return
                    else:
                        deskModelOutput = 0
                except Exception as e:
                        deskModelOutput = 0
                        print(f"Error finding desk numbers, assuming at least one doesn't exist.")

                #try grabbing the caddy number from results.
                try:
                    caddyModelOutput = results[2][0][1]
                except Exception as e:
                    caddyModelOutput = 0
                    print(f"Error finding caddy number, assuming it doesn't exist.")
            
            
                # Compare deskNumber to Desk Model output
                deskNumber = googleSheet_object.get_cell(sheet_row_counter, colDeskNum)
                # If the desk number and caddy number don't match the spreadsheet number, then mark it as a 0
                if( not isNumberinResults(deskModelOutput, deskNumber,2) and not isNumberinResults(caddyModelOutput, deskNumber,1)):
                    googleSheet_object.update_cell(sheet_row_counter, colCitizenshipScore, 0)
                    #googleSheet_object.update_cell(sheet_row_counter, colAICheck, True) #remove this in favor of saving reads/writes on the google sheets
            except: #no data exists within results
                googleSheet_object.update_cell(sheet_row_counter, colAssessmentScore, 1)
                googleSheet_object.update_cell(sheet_row_counter, colCitizenshipScore, 1)
                googleSheet_object.update_cell(sheet_row_counter, colAICheck, True)
            
            
            # Debug statements
            print(studentID)
            print(folderURL)
            print(deskNumber)
            
            
            # The google sheet object deletes files from your pc, because obviously that's the correct component
            # to be deleting things (T-T)
            #googleSheet_object.delete_temp_folder(tempFolderPath) #not today, google
            sheet_row_counter += 1
            self.ids.progress_bar_background.set_value = round(sheet_row_counter/counter*100, 2)
            self.progress_bar_value = self.ids.progress_bar_background.set_value

    #====================================================================================       
    

# main call loop for kivy to make application window
class TechTutorApp(App):
    set_value = 5
    
    def build(self):
        
        # setting window background to white
        Window.clearcolor = (49/255,51/255,56/255,1)
        return MyFloatLayout()
    
    global packet_model
    packet_model = load_model("./models/id_periodNum_model.pt", "packet")
    
    global desk_model
    desk_model = load_model("./models/desk_model.pt", "desk")
    
    global caddy_model
    caddy_model = load_model("./models/caddy_model.pt", "caddy")
    
    global models
    # must be in order of packet,desk,caddy model in tuple
    models = (packet_model,desk_model,caddy_model)

if __name__ == '__main__':
    TechTutorApp().run()
    

# helper functions to get the models
def get_packet_model():
    if packet_model is not None:
        return packet_model
    
def get_caddy_model():
    if caddy_model is not None:
        return caddy_model
    
def get_desk_model():
    if desk_model is not None:
        return desk_model