@echo off
:: ==============================
:: Initialize Local Repository and Push
:: ==============================

:: 1) Initialize Git repository in the current folder (if not already)
git init

:: 2) Stage all changes (new, modified, and deleted files)
:: - 'git add -A' ensures deleted files are also included, unlike 'git add .'
git add -A

:: 3) Commit staged changes with a message
git commit -m "Initial commit"

:: 4) Add SSH remote to GitHub
:: - 'origin' is a persistent reference to the remote repository URL
git remote remove origin 2>nul
git remote add origin git@github.com:petercl8/FlexCNN_for_Medical_Physics.git

:: 5) Rename the current branch to 'main'
git branch -M main

:: 6) Push local commits to GitHub and set upstream
:: - '-u' sets 'origin/main' as the default for future pull/push commands
git push -u origin main

echo Repository initialized and pushed to GitHub.
pause
