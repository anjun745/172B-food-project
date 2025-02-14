import os
from PIL import Image


height = 400
width = 400

rootdir = "../realsense_overhead"


#Remove unecessary files
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        if "depth" in filename:
            os.remove(filename)

#Resize images
"""for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        if '.png' in filename:
            img = Image.open(filename)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            img.save(filename)


#Move files into one directory
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        os.rename(filename, f"{rootdir}/{subdir}.png")"""
        
for subdir, dirs, files in os.walk(rootdir):
    os.remove(subdir)

