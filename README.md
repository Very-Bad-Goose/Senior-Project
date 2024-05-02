# TechTutor

![image](https://github.com/Very-Bad-Goose/Senior-Project/assets/71528875/bbd8112f-7414-4646-b828-abc9ef22a103)


## Summary
TechTutor, developed by the Byte Brigade team, is a Python-based project leveraging PyTorch for machine learning tasks. The primary objective of TechTutor is to simplify the process of grading students' homework by employing machine learning algorithms to analyze images submitted by students. The graded assignments are then conveniently uploaded to Google Drive for teachers' review. The grading process is facilitated by the input of a grading key by the teacher, which the program utilizes to assess and score the homework assignments.


# Developer Instructions

For those interested in using TechTutor, follow these instructions:

## For Prototype 2, here are the installs and steps you need: Note we are using test2.ipynb now
1. First, ensure you are in your conda environment by typing this into your IDEs terminal
        `conda activate`
   You should now see (base) followed by your hostname if you are in the conda environment
2. Use pip to install the necessary packages:
        `pip install google-auth google-cloud-vision google-api-python-client google-auth-httplib2 google-auth-oauthlib`
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
        Share the Google Sheet with the service account's email to give it access.
        (enter the email address of the service account in the share icon on the Google sheet. 
        Grant the service account “Editor” access so it can read and write to the sheet.
   
        Step 3:
        Now alter all the TODO locations to ensure your file paths/file names are set up correctly.

        Step 4:
        For now, test the prototype with the sample image provided (should be in same directory as project.
        To do this, you need to use local host, since pulling that image off the internet can cause issues.
        Do this with `python -m http.server 8000` in the terminal, then RUN the code. 

# Kivy download instructions for UI
In order to install kivy into your virtual environment for python use pip install and run

        ```python -m pip install "kivy[base]"```

More Information at: https://kivy.org/doc/stable/gettingstarted/installation.html#install-pip

# Lightning installation instructions
open a terminal an enter the following command

```python -m pip install lightning```

additional information on the library can be found at: https://github.com/Lightning-AI/pytorch-lightning

# img2pdf installation instructions
open a terminal an enter the following command

```python -m pip install img2pdf```

additional information on the library can be found at: https://pypi.org/project/img2pdf/


## Jira Timeline

For project management and tracking progress, refer to the following Jira timeline:

- **Task 1:** Project Initiation - [Link to Jira](#)
- **Task 2:** Feature Development - [Link to Jira](#)
- **Task 3:** Testing and Quality Assurance - [Link to Jira](#)
- **Task 4:** Deployment and Integration - [Link to Jira](#)
- **Task 5:** Maintenance and Support - [Link to Jira](#)

# Contributors

- Joaquim Pedroza
- Jacob Sherer
- Ryan Naveira
- Chisom Iwunze
- Collin Dunkle
- Angelo Karam
- Joshua Grindstaff
- Austin Nolte


For more information or assistance, please contact the Byte Brigade team.

---

© 2024 Byte Brigade. All rights reserved.
