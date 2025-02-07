Run the code from the terminal when in the location of the project folder after installing all the libraries.

pyinstaller --onefile -w --icon=faviconForeo.ico --add-data "emoji.json;." --add-data "Foreo_Logo.png;." Foreo_Estimator.py

Notes:
Try to avoid using public Wifi when running the code because googleapiclient may not work.