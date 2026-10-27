@echo off
rem Sheep: starts the desktop app, which starts the backend and the SHEEP engine.
set ELECTRON_RUN_AS_NODE=
cd /d "%~dp0mlmi\desktop"
if not exist "node_modules\electron\dist\electron.exe" (
  echo Installing the desktop frame, once...
  call npm install --no-audit --no-fund
)
start "" "node_modules\electron\dist\electron.exe" .
