@echo off
setlocal
cd /d "%~dp0"
g++ -std=c++17 -O2 -Wall -Wextra memory_simulator.cpp -o memory_simulator.exe
if errorlevel 1 exit /b %errorlevel%
echo Built memory_simulator.exe