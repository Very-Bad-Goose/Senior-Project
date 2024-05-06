# TechTutor

![image](https://github.com/Very-Bad-Goose/Senior-Project/assets/71528875/bbd8112f-7414-4646-b828-abc9ef22a103)


## Summary
TechTutor, developed by the Byte Brigade team, is a Python-based project leveraging PyTorch for machine learning tasks. The primary objective of TechTutor is to simplify the process of grading students' homework by employing machine learning algorithms to analyze images submitted by students. The graded assignments are then conveniently uploaded to Google Drive for teachers' review. The grading process is facilitated by the input of a grading key by the teacher, which the program utilizes to assess and score the homework assignments.

## User Interface
TechTutor has a very simple UI. It consists of a button to start, stop, and change the grading key the AI uses to grade against. The UI also has a progress bar to show the user the current grading job's completion percentage, and a window to show what current grading key is being used. It utilizes Kivy for its main UI and here is an exmaple of what one might see when using the application.
![TechTutor_Current_UI](https://github.com/Very-Bad-Goose/Senior-Project/assets/149719462/8c753d70-3487-4f2c-94a4-06c9c37d8716)


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
        Share the Google Sheet with the service account's email to give it access.
        (enter the email address of the service account in the share icon on the Google sheet. 
        Grant the service account “Editor” access so it can read and write to the sheet.
   
   Step 3:
        Now alter all the TODO locations to ensure your file paths/file names are set up correctly.

   Step 4:
        For now, test the prototype with the sample image provided (should be in same directory as project.
        To do this, you need to use local host, since pulling that image off the internet can cause issues.
        Do this with ```python -m http.server 8000``` in the terminal, then RUN the code.
   
This iteration of the AI works directly with Google Sheets, as follows:


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

- **Sprint 5:** Train AI on handwriting recognission (8/26/24 - 9/8/24)
  - Train AI to read math symbols.
  - Train AI to locate ID numbers on worksheets.
  - Make sure AI can handle imperfections in paper.
    - this includes blemishes and too blurry of pictures.
  - Train AI to locate differences in images.
  - Make the AI look at a submission and compare it against a "correct" image.
  - Make AI determine the percent difference between grading key and submission.
- **Sprint 6:** Train AI on Desk Recognition (9/9/24 - 9/22/24)
  - Train AI to handle dents in desks.
  - Train AI to make sure desk number is same as assigned desk numbers.
  - Train AI to make sure calculator is in top left of desk.
  - Train AI to make sure number on calculator matches desk number.
  - Train AI to make sure side pouches are clean.
  - Create test data sets.     
- **Sprint 7:** Train AI on Other Submissions (9/23/24 - 10/6/24)
  - Test AI on different types of assignments and different grading keys.
  - Add ability for user to test if AI can read grading key correctly.
  - Test submissions on different grading keys that are not correct.
  - Implement a threashold for how good the image quality needs to be for AI to grade assignment.        
- **Sprint 8:** UI Finishing Touches (10/7/24 - 10/20/24)
  - Progress bar update with regards to current AI job.
  - Make Start button start AI model.
  - Make Stop button stop AI model.
  - Make it so the grading key chosen by UI is the one being used by AI model.
  - Add logging feature of AI to see why it made the decisions it did.
- **Sprint 9:** Testing - Get Accuracy > 95% (10/21/24 - 11/3/24)
- **Sprint 10:** Final Testing (11/4/24 - 11/17/24)

Link to Jira Backlog: https://bad-goose.atlassian.net/jira/software/projects/PROJECT01/boards/2/backlog
# Testing

# Deployment

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
