from PIL import Image
import os
import sys

print ""
print "Prepare Celeb-A Dataset! (1. Crop the images. 2. Generate a train list file.)"
print ""
print "-------------------------------------------------------------------------------"

current_path = os.getcwd()
celebA_path = ""
celebA_cropped_path = ""
print "The current path containing this python file is: " + current_path
if len(sys.argv) == 1:
    print "Please give the path of original Celeb-A dataset!"
    exit(0)
elif len(sys.argv) > 1:
    print "The path of original Celeb-A dataset is: " + str(sys.argv[1])
    celebA_path = sys.argv[1]
    celebA_cropped_path = os.path.dirname(celebA_path) + os.sep + "Cropped"     #To avoid crop the generated images again if this parameter is not provided
    if len(sys.argv) > 2:
        print "The path of cropped Celeb-A dataset will be: " + str(sys.argv[2])
        celebA_cropped_path = sys.argv[2]
    else:
        print "The path of cropped Celeb-A dataset will be defult, set as: " + celebA_cropped_path

if os.path.exists(celebA_cropped_path):
    print "The path of cropped Celeb-A dataset exists."
else:
    print "The path of cropped Celeb-A dataset doesn't exist! I will create it now!"
    os.makedirs(celebA_cropped_path)
print "-------------------------------------------------------------------------------"

training_list_file = os.path.join(celebA_cropped_path, "celebA.txt")
list_file = open(training_list_file, 'w')
total_image_num = 0
x1, y1 = 30, 40
cropped_box = (x1, y1, x1 + 138, y1 + 138)

for parent,dirnames,filenames in os.walk(celebA_path):
    for filename in filenames:
        if filename.endswith(".jpg"):
            total_image_num += 1
            #print "parent is:" + parent
            #print "filename is:" + filename
            image_path_and_name = os.path.join(parent,filename)
            print "the full name of the file is: " + image_path_and_name
            input_image = Image.open(image_path_and_name)
            #input_image.show()
            cropped_image = input_image.crop(cropped_box)
            #cropped_image.show()
            scaled_cropped_image = cropped_image.resize((64, 64))
            #scaled_cropped_image.show()
            save_result_image_path_and_name = os.path.join(celebA_cropped_path,filename)
            scaled_cropped_image.save(save_result_image_path_and_name, 'jpeg')
            list_file.writelines(save_result_image_path_and_name)
            list_file.writelines(" 1" + "\n")   #Must add label to list file
print "There are " + str(total_image_num) + " images are finished with cropping and scaling operations!"
list_file.close()