@echo off
:: ===== Start Session =====

echo Checking for local changes...

git diff-index --quiet HEAD --
if %errorlevel% neq 0 (
    echo WARNING: You have uncommitted local changes that will be discarded.
    pause
)

echo Resetting repository to match GitHub...

git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd

echo Repository is now synced with GitHub.
