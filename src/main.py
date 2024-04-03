import kivy
import kivy.properties as kyProps
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window


# loading kv language file
Builder.load_file('TechTutor.kv')

# disbaling touch screen emulation on mouse
Config.set("input","mouse","mouse,disable_multitouch")

# custom layout to hold all UI elements using a KV file for UI elements
class MyFloatLayout(FloatLayout):
    
    # method for when start button is pressed
    def start_press(self):
        print("here")
    
    # method for when stop button is pressed
    def stop_press(self):
        print("here")
    
    # method for when change grade key button is pressed
    def change_key_button(self):
        print("here")
    
    # method for when pause button is pressed    
    def pause_press(self):
        print("here")
    pass

# main call loop for kivy to make application window
class TechTutorApp(App):
    def build(self):
        
        # setting window background to white
        Window.clearcolor = (130/255,129/255,127/255,1)
        return MyFloatLayout()
    
if __name__ == '__main__':
    TechTutorApp().run()