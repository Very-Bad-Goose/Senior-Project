import img2pdf
from PIL import Image
from pathlib import Path
import os
 
#demo code on how the program converts an image into a pdf
"""
# storing image name
img_path = "input.jpg"
#get current path
path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) 
#add the extra image path to the path and 
#convert it to a string that can be used by the rest of the program
#will need testing on non-windows systems
path = path + '/' + img_path
p = repr(path)
a = p.replace('\\\\','/')
a = a.replace('WindowsPath(','')
a = a.replace(')','')
a = a.replace('\'','')
print(a)

# storing pdf name
pdf_name = "output.pdf"
 
# opening image
image = Image.open(a)

# converting into chunks using img2pdf
pdf_bytes = img2pdf.convert(image.filename)
 
# opening or creating pdf file
file = open(pdf_name, "wb")
 
# writing pdf files with chunks
file.write(pdf_bytes)
 
# closing image file
image.close()
 
# closing pdf file
file.close()
 
# output
print("Successfully made pdf file")"""



#convert provided image (with full path) into pdf format
#name is the desired name for output pdf
def toPDF(image_path, name):
    import img2pdf
    image = Image.open(image_path)
    file = open(name, "wb") #create a pdf document in current folder with name given by pdf name and allows for writing binary data into it
    file.write(img2pdf.convert(image.filename))#was outputting into parent folder for some reason
    image.close()
    file.close()
    print(name)

"""
#incomplete, will require additional libraries (and setting a new path variable) if we plan to add this as a feature
#convert pdf (using path provided) back into an image of a given type assuming the pdf was created by this program
#name is the desired name for the output image
def toImage(pdf_path, name):
    from pdf2image import convert_from_path
    image = convert_from_path(pdf_path) #assumes pdf 
    print(Path(pdf_path).parent)
"""

#test code. will need pathing to not require absolute pathing
toPDF('C:/Users/Username/Documents/GitHub/Senior-Project/src/input.jpg','output.pdf')
