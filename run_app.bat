@echo off
REM Activate the environment (adjust path to client's)
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate base

REM Checks if there is an existing environment and creates one if there is not
conda env list | findstr "cputorch" >nul
if errorlevel 1 (
    echo Did not detect existing environment named "cputorch"
    echo Rerun bat file when installation is done
    echo Making new environment...
    conda env create -f cpu_environment.yml
)

call conda activate cputorch

REM Run the Python script
python "src\main.py"