@echo off
setlocal

REM Variables
set "MinicondaInstallerUrl=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "InstallerPath=%TEMP%\\MinicondaInstaller.exe"
set "InstallDir=%USERPROFILE%\\Miniconda3"

REM Check if Miniconda is already installed
if exist "%InstallDir%\\condabin\\conda.bat" (
    echo Miniconda is already installed at %InstallDir%.\
    pause
    goto :EOF
)

REM Download the Miniconda installer
echo Downloading Miniconda installer...
powershell -Command "Invoke-WebRequest -Uri '%MinicondaInstallerUrl%' -OutFile '%InstallerPath%'"
if not exist "%InstallerPath%" (
    echo Failed to download the Miniconda installer.
    exit /b 1
)

REM Install Miniconda silently
echo Installing Miniconda...
"%InstallerPath%" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%InstallDir%
if %ERRORLEVEL% neq 0 (
    echo Miniconda installation failed.
    exit /b 1
)

REM Cleanup
del "%InstallerPath%"
echo Installer removed.

REM Verify installation
if exist "%InstallDir%\\condabin\\conda.bat" (
    echo Miniconda installation completed successfully.
) else (
    echo Miniconda installation failed.
    exit /b 1
)

endlocal
pause