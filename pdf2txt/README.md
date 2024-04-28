# pdf2txt script.

This script converts PDFs to a one PDF per line, Tab separated `[{doc title}{Tab}{doc text}]` text file using the PyPDF library.

Run `pip install -r requirements.txt` to install all required packages.


# Scrub manual copy/pasted text script.

This script scrubs and reformats copy/pasted text from PDFs to a one text file per line, Tab separated `[{doc title}{Tab}{doc text}]` text file. It does not require any external libraries.


# Recommended steps.

## pdf2txt.py

After placing the PDFs you wish to convert to text in the local `documents` folder, run the script using:

`python3 pdf2txt.py`

Results can be found in the `output.txt` file. Running this script multiple times will append subsequent results to the end of the file.

## OR

## scrub_manual_input.py

For each document/PDF, copy into a new plain text file the title of the document in the first line. Then, the rest of the desired text from the document on subsequent lines. The script will automatically handle `[]`-type references and `\n` in the text (see examples in `copy_paste_text` folder). Once all desired text is copy/pasted to a text file (one text file per document), and placed in `copy_paste_text`, run the script using:

`python3 scrub_manual_input.py`

Results can be found in the `scrubbed_output.txt` file.