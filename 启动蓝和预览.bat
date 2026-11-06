@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if not exist "%~dp0land_ccs_preview.py" goto missing_script

where python.exe >nul 2>&1
if not errorlevel 1 (
    python.exe -c "import tkinter, numpy, matplotlib" >nul 2>&1
    if not errorlevel 1 (
        where pythonw.exe >nul 2>&1
        if not errorlevel 1 (
            start "" pythonw.exe "%~dp0land_ccs_preview.py" %*
            goto done
        )
        python.exe "%~dp0land_ccs_preview.py" %*
        goto done
    )
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    py.exe -3 -c "import tkinter, numpy, matplotlib" >nul 2>&1
    if not errorlevel 1 (
        where pyw.exe >nul 2>&1
        if not errorlevel 1 (
            start "" pyw.exe -3 "%~dp0land_ccs_preview.py" %*
            goto done
        )
        py.exe -3 "%~dp0land_ccs_preview.py" %*
        goto done
    )
)

echo Python with tkinter, numpy and matplotlib is required.
echo Please check the Python installation and dependencies.
pause
popd
exit /b 1

:missing_script
echo land_ccs_preview.py was not found beside this launcher.
pause
popd
exit /b 1

:done
popd
endlocal
exit /b 0
