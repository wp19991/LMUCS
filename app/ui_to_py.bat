call C:\miniconda3\Scripts\activate.bat C:\miniconda3
pyrcc5 res/app.qrc -o res/app_rc.py
python -m PyQt5.uic.pyuic ui/main_window.ui -o ui/main_window.py --import-from=res