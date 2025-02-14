import os

rootdir = "../realsense_overhead"


#Remove unecessary files
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        if "depth" in filename:
            os.remove(filename)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        print(filename)
