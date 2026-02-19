@echo off
:: ==============================
:: One-Time Machine Setup (SSH + Git + Google Drive Path)
:: ==============================

:: --- GitHub setup ---
:: Prompt for GitHub username
set /p GITHUB_USER="Enter your GitHub username: "

:: Prompt for GitHub email
set /p GITHUB_EMAIL="Enter your GitHub email: "

:: Set global Git identity
git config --global user.name "%GITHUB_USER%"
git config --global user.email "%GITHUB_EMAIL%"

:: Generate SSH key if it doesn't exist

if not exist "%USERPROFILE%\.ssh" (
    mkdir "%USERPROFILE%\.ssh"
)

if not exist "%USERPROFILE%\.ssh\id_ed25519" (
    echo Generating new SSH key...
    ssh-keygen -t ed25519 -f "%USERPROFILE%\.ssh\id_ed25519" -N ""
) else (
    echo SSH key already exists.
)

:: Show public key for GitHub
echo =========================
echo Copy the following key to GitHub (Account Settings → SSH and GPG keys):
type "%USERPROFILE%\.ssh\id_ed25519.pub"
echo =========================
pause

:: Test SSH connection to GitHub
echo Testing SSH connection to GitHub...
ssh -T git@github.com
pause


echo Machine setup complete!
pause