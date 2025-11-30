@echo off
REM ===============================
REM Git rollback helper
REM ===============================

REM --- Config ---
setlocal enabledelayedexpansion

REM The branch to operate on
set BRANCH=main

REM --- Step 0: Disable Git pager for script ---
set GIT_PAGER=

REM --- Step 1: Make sure we're on the target branch ---
git checkout %BRANCH%
if errorlevel 1 (
    echo ❌ Failed to checkout %BRANCH%.
    pause
    exit /b
)

REM --- Step 2: Backup current branch ---
set BACKUP_BRANCH=backup_before_rollback_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
echo Creating backup branch !BACKUP_BRANCH! ...
git branch !BACKUP_BRANCH!
if errorlevel 1 (
    echo ❌ Failed to create backup branch.
    pause
    exit /b
)
git push origin !BACKUP_BRANCH!
echo ✅ Backup branch created and pushed.

REM --- Step 3: Show last 20 commits ---
echo.
echo Last 20 commits on %BRANCH%:
git --no-pager log --oneline -20
echo.

REM --- Step 4: Ask for commit hash to rollback to ---
set /p TARGET_HASH=Enter the commit hash to rollback to: 

REM --- Step 5: Reset local branch ---
git reset --hard %TARGET_HASH%
if errorlevel 1 (
    echo ❌ Failed to reset branch.
    pause
    exit /b
)
echo ✅ Branch reset to %TARGET_HASH%.

REM --- Step 6: Push reset to remote ---
echo.
echo Do you want to force push this rollback to remote? (Y/N)
set /p PUSH_CONFIRM=
if /i "%PUSH_CONFIRM%"=="Y" (
    git push --force origin %BRANCH%
    if errorlevel 1 (
        echo ❌ Failed to push rollback to remote.
        pause
        exit /b
    )
    echo ✅ Rollback pushed to remote.
) else (
    echo Rollback not pushed. You can push manually later if desired.
)

echo.
echo ✅ Rollback complete.
pause
