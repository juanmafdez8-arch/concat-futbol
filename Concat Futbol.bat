@echo off
rem Abre el Concat modificado (fork futbol) con las DLL de FFmpeg y sherpa-onnx en el PATH.
set "D=D:\concat-dev"
set "PATH=%D%\ffmpeg-n8.1-latest-win64-gpl-shared-8.1\bin;%D%\sherpa\sherpa-onnx-v1.13.7-win-x64-shared-MT-Release-lib\lib;%PATH%"
start "" "%D%\target\release\quick\concat.exe"
