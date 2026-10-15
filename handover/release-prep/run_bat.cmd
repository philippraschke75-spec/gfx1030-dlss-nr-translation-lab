@echo off
rem usage: run_bat.cmd <bat> <answers> <log>   (cwd = fake game folder)
cd /d E:\dlss5-v0.3-prep\fakegame
call "E:\dlss5-v0.3-prep\fakegame\%~1" < "%~2" > "%~3" 2>&1
