@echo off
:: ==============================
:: Close Session: Commit & Push All Changes to Repository
:: ==============================

:: Ask if user wants to push remaining changes
set /p pushfinal=Push any remaining changes before closing? (y/n): 
if /i "%pushfinal%"=="y" (

    :: Stage all changes
    git add -A

    :: Prompt for commit message
    set /p msg=Enter commit message: 
    if "%msg%"=="" set msg=Final update

    :: Only commit if there are staged changes
    git diff --cached --quiet
    if %errorlevel%==1 (
        git commit -m "%msg%"
    ) else (
        echo No changes to commit.
    )

    :: Push commits to the main branch
    git push origin main
)

echo Session closed.
pause
