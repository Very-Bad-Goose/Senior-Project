# TechTutor

![image](https://github.com/Very-Bad-Goose/Senior-Project/assets/71528875/bbd8112f-7414-4646-b828-abc9ef22a103)


## Summary
TechTutor, developed by the Byte Brigade team, is a Python-based project leveraging PyTorch for machine learning tasks. The primary objective of TechTutor is to simplify the process of grading students' homework by employing machine learning algorithms to analyze images submitted by students. The graded assignments are then conveniently uploaded to Google Drive for teachers' review. The grading process is facilitated by the input of a grading key by the teacher, which the program utilizes to assess and score the homework assignments.





## User Interface
TechTutor has a very simple UI. It consists of a few buttons and text boxes. The Sheet URL text box is where the user inputs the URL for the Google Sheet and the Save Sheet ID button will save the URL. The Number of Packet Pages text box is where the user enters the expected amount of images that a student should have for a submission and will only accept numbers larger than 0. You save the number in this text box by clicking the "Save Number". The "Change Credentials" button allows the user to change the Json file that is acquired from a Google Service Account. The "Start" button is the button you will push to run the AI models against the values in the sheet. The "Stop" button will stop the application while it is running.
![image](https://github.com/user-attachments/assets/4fbade32-7dc4-43e9-9bc9-babdcff9fdb7)



# How to Download, Setup, and Run

### Prerequisites

#### Install Miniconda
**Option 1: Manual Installation**
- Download Miniconda for your platform from [Miniconda Downloads](https://docs.conda.io/en/latest/miniconda.html).
- Follow the installer instructions for your operating system.
- Verify installation by running:
  ```
  conda --version
  ```
**Option 2: Automated Installation**
- After cloning the repository use the provided InstallMiniconda.bat script.
- Navigate to the directory containing the script.
  ```
  InstallMiniconda.bat
  ```
- After the script completes, verify installation by running:
  ```
  conda --version
  ```
### Step 1: Clone the Repository
- Open a terminal and navigate to your desired folder.
  ```
  git clone https://github.com/Very-Bad-Goose/Senior-Project.git
  ```
- Navigate into the project directory:
  ```
  cd Senior-Project
  ```
### Step 2: Run the Deployment Script
- Execute the provided deployment script:
  ```
  run_app.bat
  ```
#### The script will:
- Use the provided .yml file to create and configure the necessary Conda environment.
- Install all dependencies required for the project.

### Step 3: Run the Program
- Inside the project directory, start the program by running:
  ```
  python main.py
  ```

# Developer Instructions

For those interested in using TechTutor, follow these instructions: (Note we are using test2.ipynb now)
1. First, ensure you are in your conda environment by typing this into your IDEs terminal
        `conda activate`
   You should now see (base) followed by your hostname if you are in the conda environment
2. Use pip to install the necessary packages:
   ```pip install google-auth google-cloud-vision google-api-python-client google-auth-httplib2 google-auth-oauthlib```
3. Create a service account to get an API key and JSON file:

   Step 1:
        Go to the Google Cloud Console.
        At the top left, click on "No project selected" and create a new project
        Navigate to APIs & Services on the Dashboard to the left and click ENABLE APIS AND SERVICES to search for Google Sheets API, then hit enable.
        In the menu on the left, click Credentials, then at the top of the screen, click create credentials then
        click Create Service Account.
        Enter a name and description for the service account. Click Create.
        You don’t need to grant this account any role on the Cloud project for this purpose, so click Done.

   Step 2:
        Create Keys for the Service Account.
        Click on the service account you just made in the list. (take note of the email/copy it for later)
        Go to the Keys tab.
        Click Add Key, choose Create new key, and select JSON. Click Create.
        This action will download a JSON file to your computer (move to the Project directory)
        Share the Google Sheet with the service account's email to give it access
        (enter the email address of the service account in the share icon on the Google sheet). 
        Grant the service account “Editor” access so it can read and write to the sheet.
   
   Step 3:
        Now alter all the TODO locations to ensure your file paths/file names are set up correctly.

   Step 4:
        For now, test the prototype with the sample image provided (should be in same directory as project).
        To do this, you need to use local host, since pulling that image off the internet can cause issues.
        Do this with ```python -m http.server 8000``` in the terminal, then RUN the code.
   
This iteration of the AI works directly with Google Sheets, as follows:
This is an example of a input image, which you would place a link to (or path to) in a Google Sheet cell
![image](https://github.com/Very-Bad-Goose/Senior-Project/blob/main/src/assests/dogcat.png)
Here is an example of what you can set the Google Sheets to look like where you provide the link or path to the input
image, followed by the cell/column where the AI can output what it "saw". Make sure to specify these cell numbers in your code.
![image](https://github.com/Very-Bad-Goose/Senior-Project/blob/main/src/assests/googleSheetsView.png)

# Training Data Fetch
1. Modify fetch_google_drive
   
   * Change `SERVICE_ACCOUNT_FILE` to your .json file (The one you made in developer instructions)
   * Change `SUBMISSION_FOLDER_ID` = to the Folder ID of the submission folder of the target drive. You can find the folder ID in the url when viewing the folder (may not need to changed).
   * ![Folder ID](https://github.com/user-attachments/assets/f1390b0c-151a-46b4-a2c0-5e9eba3c899e)
   * Change `DATA_PATCH` to the target folder where the test and training data will be created

2. Add the service account email to the submission folder share permission on the google drive
3. Make sure to add the test.txt and train.txt files to your test and train folder. These text files keep track of which folders are for testing and for training so we can have consistency.


# Kivy Download Instructions for UI
In order to install kivy into your virtual environment for python use pip install and run

```python -m pip install "kivy[base]"```

More Information at: https://kivy.org/doc/stable/gettingstarted/installation.html#install-pip

Additional information on the library can be found at: https://pypi.org/project/img2pdf/

Link to Jira Backlog: https://bad-goose.atlassian.net/jira/software/projects/PROJECT01/boards/2/backlog
# Testing

In order to run one of the several PyTest test suites we have built for this project, please run ```pytest test_file_name``` in terminal for a specific test file. For example, to run src/test_image_blur_detection.py, type ```pytest test_image_blur_detection.py``` into terminal.

# Contributors

- Joaquim Pedroza: Email address: joaquimpedroza@csus.edu Phone number: 530-957-4456 
- Jacob Sherer: Email address: jacobsherer@csus.edu Phone number: 916-622-8684 
- Ryan Naveira: Email address: rnaveira@csus.edu Phone number: 559-970-4024 
- Chisom Iwunze: Email address: chisomiwunze@csus.edu Phone number: 925-848-6248
- Collin Dunkle: Email address: cdunkle@csus.edu Phone number: 916-865-8355 
- Angelo Karam: Email address: akaram@csus.edu Phone number: 916-809-4903 
- Joshua Grindstaff: Email address: joshuagrindstaff@csus.edu Phone number: 916-647-7548 
- Austin Nolte: Email address: arnolte@csus.edu Phone number: 209-352-7436 


For more information or assistance, please refer to attached manuals.

---

© 2024 Byte Brigade. All rights reserved.
