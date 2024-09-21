from pdf2image import convert_from_path
from PIL import Image

# Pfad zur PDF-Datei und der gewünschten Ausgabedatei
pdf_path = 'MOT_ML.pdf'
output_image_path = 'preview.png'

# Pfad zu poppler
poppler_path = r'C:\Program Files\poppler\Library\bin'

# Die erste Seite der PDF in ein Bild umwandeln
pages = convert_from_path(pdf_path, 300, first_page=0, last_page=1, poppler_path=poppler_path)

# Die erste Seite als Bild speichern
pages[0].save(output_image_path, 'PNG')
