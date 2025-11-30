@echo off
:: ===== Checkpoint Update (safe for single-developer workflow) =====

:: Stage all changes
git add .

:: Prompt for commit message
set /p msg=Enter checkpoint commit message: 
if "%msg%"=="" set msg=Checkpoint update

:: Commit (ignore if nothing changed)
git commit -m "%msg%" 2>nul || echo Nothing to commit.

:: Force push local main to remote
git push --force origin main

echo ✅ Checkpoint changes pushed to GitHub.
pause
