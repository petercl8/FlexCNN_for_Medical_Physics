@echo off

:: --- Git operations ---
git add -A
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Quick automatic update"
) else (
    echo No changes to commit.
)

:: Force push to remote to avoid pull/reject issues
git push --force origin main

echo ✅ Update complete!
pause
