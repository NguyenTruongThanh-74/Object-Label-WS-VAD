@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%~1"=="push" goto push_action
if /I "%~1"=="pull" goto pull_action

echo Usage: scripts\sync_git.bat push [commit message]
echo        scripts\sync_git.bat pull
exit /b 2

:prepare
pushd "%~dp0.." || exit /b 1
git rev-parse --show-toplevel >nul 2>&1
if errorlevel 1 (
  echo Error: this script is not inside a Git repository.
  popd
  exit /b 1
)
set "BRANCH="
for /f "delims=" %%B in ('git branch --show-current') do set "BRANCH=%%B"
if not defined BRANCH (
  echo Error: detached HEAD; check out a branch before syncing.
  popd
  exit /b 1
)
echo Repository: %CD%
echo Branch: !BRANCH!
git status --short --branch
exit /b 0

:pull_action
call :prepare
if errorlevel 1 exit /b 1
set "HAS_CHANGES="
for /f "delims=" %%S in ('git status --porcelain') do set "HAS_CHANGES=1"
if defined HAS_CHANGES (
  echo Error: working tree is not clean. Commit or stash changes before pulling.
  popd
  exit /b 1
)
git pull --rebase
if errorlevel 1 (
  echo Pull stopped. Resolve conflicts, then continue the rebase or run: git rebase --abort
  popd
  exit /b 1
)
git status --short --branch
popd
exit /b 0

:push_action
call :prepare
if errorlevel 1 exit /b 1
choice /C YN /M "Stage all non-ignored changes in this repository"
if errorlevel 2 goto canceled_before_stage
git add -A
if errorlevel 1 goto failed

git diff --cached --quiet
if errorlevel 1 goto commit_changes
goto push_changes

:commit_changes
git diff --cached --stat
choice /C YN /M "Commit staged changes and push to origin/!BRANCH!"
if errorlevel 2 goto canceled_after_stage
set "MESSAGE=%~2"
if not defined MESSAGE set "MESSAGE=Update project"
git commit -m "%MESSAGE%"
if errorlevel 1 goto failed

:push_changes
git push --set-upstream origin "!BRANCH!"
if errorlevel 1 goto failed
git status --short --branch
popd
exit /b 0

:canceled_before_stage
echo Canceled; nothing was staged.
popd
exit /b 0

:canceled_after_stage
echo Canceled; changes remain staged and were not pushed.
popd
exit /b 0

:failed
echo Git operation failed. Review the output above.
popd
exit /b 1
