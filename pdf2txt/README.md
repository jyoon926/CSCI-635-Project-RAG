# pdf2txt script.

This script converts PDFs to a one PDF per line, Tab separated `[{doc title}{Tab}{doc text}]` text file using the PyPDF library.

Run `pip install -r requirements.txt` to install all required packages.



# Recommended steps.

After placing the PDFs you wish to convert to text in the local `documents` folder, run the script using:

`python3 pdf2txt.py`

Results can be found in the `output.txt` file. Running this script multiple times will append subsequent results to the end of the file.