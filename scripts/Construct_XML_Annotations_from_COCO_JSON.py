"""
I'm expecting you have openCV, numpy, and pandas
since you're about to run some ML algorithms that requires most of them.

If you've run this script before, then I will be producing a script to look
through all the subdirectories and restore the files to their original location.
"""
import json
from os import path, environ, mkdir
import argparse
from shutil import move, copyfile
from cv2 import imread
from lxml import etree, objectify
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    description='Construct_XML_Annotations_from_JSON')
parser.add_argument('-y', '--year',
                    help="The year to append to train and val directories",
                    type=int,
                    required=True)
parser.add_argument('-t', '--type',
                    help="The type of imageset, 'train' or 'val'",
                    choices=["train", "val"],
                    type=str, required=True)
parser.add_argument('-c', '--container', help="'instances'",
                    choices=["instances"], type=str, default="instances")
parser.add_argument('-s', '--split',
                    help="If you want to further split the training set", type=float)
parser.add_argument('-l', '--label',
                    help="Generate a new labels.txt including this label. Repeatable",
                    action='append', type=str)
parser.add_argument(
    '-p', '--copy', help="Copy files don't move them.", action="store_true")
args = parser.parse_args()

# import numpy as np
YEAR = args.year
TYPEPREFIX = args.type  # "train" or "val"
# For example this default expects: $HOME/data/coco/Images/train2017
# and $HOME/data/coco/Annotations
CONTAINER = args.container
# Further splits data if you are not going on to do the Test stage of COCO
# Creates train.txt and minival.txt with image locations and annotation
# locations if type is "train"
if args.split:
    TRAIN_SPLIT = args.split
# Helpful if you consider running this program several times
# DANGER: Until the companion program is created, you can just delete the train_minusminival20xy/ directory
# ***___ONLY___*** if you've used the copy the last time. If you used move, then copy those files back in
# to your PATH_TO_IMAGES
COPY_FILES = args.copy

# Will create annotations/val or annotations/train
OUTPUT_XML_BASE_DIR = "%s/data/coco/Annotations/" % (environ['HOME'])
# sometimes we just have everything in one directory as silly as that sounds
INPUT_JSON_BASE_DIR = "%s/data/coco/Annotations/" % (environ['HOME'])
ROOT_COCO = "%s/data/coco/" % (environ['HOME'])
PATH_TO_IMAGES = "%s/data/coco/Images/%s%d/" % (
    environ['HOME'], TYPEPREFIX, YEAR)
PATH_TO_IMAGES = "%s/data/coco/Images/%s%d/" % (
    environ['HOME'], TYPEPREFIX, YEAR)
ROUND_COORDS = True

with open(INPUT_JSON_BASE_DIR + CONTAINER + '_' + TYPEPREFIX + str(YEAR) + '.json', 'r') as jsfile:
    jsonfile = json.load(jsfile)
    # To write labels.txt with appropriate labels, uncomment this section
    if args.label:
        i = 0
        with open('labels.txt', mode='w') as labels:
            for row in jsonfile['categories']:
                # Remember to change supercategories or names depending on what you want. If you want everything
                # Just take out the if line
                if row['name'] in args.label:
                    i += 1
                    labels.write(str(row['id']) + "," +
                                 str(i) + ',' + row['name'] + '\n')
    label_ids = []
    with open('labels.txt') as labels:
        labels_file = labels.readlines()
        for label in labels_file:
            # Labelid is the original COCO label for the object
            labelid = label.rstrip().split(',')[0]
            # In case we get a spare empty string at the end
            if not labelid == "":
                label_ids.append(int(labelid))
print(label_ids, "xxx")
new_set = []
for annotation in jsonfile['annotations']:
    if annotation['category_id'] in label_ids:
        new_set.append(annotation)
if len(new_set) == 0:
    print("Perhaps you entered a label incorrectly, since there are no results which match.")
    exit(0)
print(len(new_set), "xxx")
df = pd.DataFrame(data=new_set)
df = df.sort_values(['image_id', 'category_id'])
grouped = df.groupby(['image_id'])

if not path.isdir(OUTPUT_XML_BASE_DIR + TYPEPREFIX):
    mkdir(OUTPUT_XML_BASE_DIR + TYPEPREFIX)
IMAGE_NAMES = []
ANN_DIR = '%sAnnotations/%s%d/' % (ROOT_COCO, TYPEPREFIX, YEAR)
if not path.isdir(ANN_DIR):
    mkdir(ANN_DIR)
for image in grouped:
    imagename = "%012d" % (image[0])
    IMAGE_NAMES.append(imagename)
    if path.isfile(ANN_DIR + imagename + ".xml"):
        continue  # Don't overwrite, unless we force it to
    image_file = imread(PATH_TO_IMAGES + imagename + '.jpg')
    E = objectify.ElementMaker(annotate=False)
    img_annotation = E.annotation(
        E.folder(TYPEPREFIX),
        E.filename(imagename),
        E.source(
            E.database('MS COCO 2017'),
        ),
        E.size(
            E.width(image_file.shape[1]),
            E.height(image_file.shape[0]),
            E.depth(3),
        ),
        E.segmented(0)
    )
    for row in image[1].iterrows():
        if row[1]['category_id'] in label_ids:
            objectNode = E.object(
                E.name(str(row[1]['category_id'])),
                E.pose("Unspecified"),
                E.truncated("0"),
                E.difficult("0"),
                E.bndbox(
                    E.xmin(
                        str(int(round(row[1]['bbox'][0]))) if ROUND_COORDS else str(int(row[1]['bbox'][0]))),
                    E.ymin(
                        str(int(round(row[1]['bbox'][1]))) if ROUND_COORDS else str(int(row[1]['bbox'][1]))),
                    E.xmax(str(int(round(row[1]['bbox'][0] + row[1]['bbox'][2])))
                           if ROUND_COORDS else str(int(row[1]['bbox'][0] + row[1]['bbox'][2]))),
                    E.ymax(str(int(round(row[1]['bbox'][1] + row[1]['bbox'][3])))
                           if ROUND_COORDS else str(int(row[1]['bbox'][1] + row[1]['bbox'][3]))),
                ),
            )
        img_annotation.append(objectNode)
    xml_pretty = etree.tostring(img_annotation, pretty_print=True)
    with open(ANN_DIR + imagename + ".xml", 'wb') as ann_file:
        ann_file.write(xml_pretty)

print("finished with xmls, now moving or copying")
if TYPEPREFIX == 'train':
    if TRAIN_SPLIT and TRAIN_SPLIT < 1.0:
        TRAIN_DIR = 'Images/train_minusminival%d/' % (YEAR)
        TRAIN_FULL_DIR = '%s%s' % (ROOT_COCO, TRAIN_DIR)
        TRAIN_ANN_DIR = 'Annotations/train_minusminival%d/' % (YEAR)
        TRAIN_FULL_ANN_DIR = '%s%s' % (ROOT_COCO, TRAIN_ANN_DIR)
        MINIVAL_DIR = 'Images/minival%d/' % (YEAR)
        MINIVAL_FULL_DIR = '%s%s' % (ROOT_COCO, MINIVAL_DIR)
        MINIVAL_ANN_DIR = 'Annotations/minival%d/' % (YEAR)
        MINIVAL_FULL_ANN_DIR = '%s%s' % (ROOT_COCO, MINIVAL_ANN_DIR)
        if not path.isdir(TRAIN_DIR):
            mkdir(TRAIN_FULL_DIR)
            mkdir(TRAIN_FULL_ANN_DIR)
        if not path.isdir(MINIVAL_DIR):
            mkdir(MINIVAL_FULL_DIR)
            mkdir(MINIVAL_FULL_ANN_DIR)
        SELECTED_FOR_MINIVAL = []
        while len(SELECTED_FOR_MINIVAL) < (1.0 - TRAIN_SPLIT) * len(IMAGE_NAMES):
            RANDOM_IDX = np.random.randint(0, high=len(IMAGE_NAMES), size=1)
            while RANDOM_IDX in SELECTED_FOR_MINIVAL:
                RANDOM_IDX = np.random.randint(
                    0, high=len(IMAGE_NAMES), size=1)
            SELECTED_FOR_MINIVAL.append(int(RANDOM_IDX))
        SELECTED_FOR_TRAIN = sorted(
            list(set(list(range(len(IMAGE_NAMES)))).difference(SELECTED_FOR_MINIVAL)))
        with open('train2.txt', 'w') as train_file:
            for idx in SELECTED_FOR_TRAIN:
                if COPY_FILES:
                    copyfile(
                        PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                        TRAIN_FULL_DIR + IMAGE_NAMES[idx] + ".jpg")
                    copyfile(
                        ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                        TRAIN_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml")
                else:
                    move(
                        PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                        TRAIN_FULL_DIR + IMAGE_NAMES[idx] + ".jpg")
                    move(
                        ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                        TRAIN_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml")
                train_file.write(
                    "/" + TRAIN_DIR + IMAGE_NAMES[idx] + ".jpg /" + TRAIN_ANN_DIR + IMAGE_NAMES[idx] + ".xml\n")
        with open('minival2.txt', 'w')as minival_file:
            for idx in SELECTED_FOR_MINIVAL:
                if COPY_FILES:
                    copyfile(
                        PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                        MINIVAL_FULL_DIR + IMAGE_NAMES[idx] + ".jpg"
                    )
                    copyfile(
                        ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                        MINIVAL_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml"
                    )
                else:
                    move(
                        PATH_TO_IMAGES + IMAGE_NAMES[idx] + ".jpg",
                        MINIVAL_FULL_DIR + IMAGE_NAMES[idx] + ".jpg"
                    )
                    move(
                        ANN_DIR + IMAGE_NAMES[idx] + ".xml",
                        MINIVAL_FULL_ANN_DIR + IMAGE_NAMES[idx] + ".xml"
                    )
                minival_file.write(
                    "/" + MINIVAL_DIR + IMAGE_NAMES[idx] + ".jpg /" + MINIVAL_ANN_DIR + IMAGE_NAMES[idx] + ".xml\n")
    else:
        with open('train2.txt', 'w') as train_file:
            IMG_RELATIVE = '/Images/train%d/' % (YEAR)
            TRAIN_ANN_DIR = 'Annotations/train%d/' % (YEAR)
            TRAIN_FULL_ANN_DIR = "%s%s" % (ROOT_COCO, TRAIN_ANN_DIR)
            if not path.isdir(TRAIN_FULL_ANN_DIR):
                mkdir(TRAIN_FULL_ANN_DIR)
            for i in range(len(IMAGE_NAMES)):
                train_file.write(
                    IMG_RELATIVE + IMAGE_NAMES[i] + ".jpg /" + TRAIN_ANN_DIR + IMAGE_NAMES[i] + ".xml\n")
else:
    with open('val2.txt', 'w') as val_file:
        IMG_RELATIVE = '/Images/val%d/' % (YEAR)
        VAL_ANN_DIR = 'Annotations/val%d/' % (YEAR)
        VALL_FULL_ANN_DIR = '%s%s' % (ROOT_COCO, VAL_ANN_DIR)
        if not path.isdir(VAL_ANN_DIR):
            mkdir(VAL_ANN_DIR)
        for i in range(len(IMAGE_NAMES)):
            val_file.write(
                IMG_RELATIVE + IMAGE_NAMES[i] + ".jpg /" + VAL_ANN_DIR + IMAGE_NAMES[i] + ".xml\n")

"""
Example of Pascal VOC 2009 annotation XML
<annotation>
	<filename>2009_005311.jpg</filename>
	<folder>VOC2012</folder>
	<object>
		<name>diningtable</name>
		<bndbox>
			<xmax>364</xmax>
			<xmin>161</xmin>
			<ymax>301</ymax>
			<ymin>200</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>298</xmax>
			<xmin>176</xmin>
			<ymax>375</ymax>
			<ymin>300</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Rear</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>432</xmax>
			<xmin>273</xmin>
			<ymax>339</ymax>
			<ymin>205</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>413</xmax>
			<xmin>297</xmin>
			<ymax>375</ymax>
			<ymin>268</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>465</xmax>
			<xmin>412</xmin>
			<ymax>273</ymax>
			<ymin>177</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Left</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>463</xmax>
			<xmin>427</xmin>
			<ymax>329</ymax>
			<ymin>225</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Left</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>186</xmax>
			<xmin>85</xmin>
			<ymax>374</ymax>
			<ymin>250</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>232</xmax>
			<xmin>74</xmin>
			<ymax>307</ymax>
			<ymin>175</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>273</xmax>
			<xmin>233</xmin>
			<ymax>200</ymax>
			<ymin>148</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Frontal</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>369</xmax>
			<xmin>313</xmin>
			<ymax>235</ymax>
			<ymin>165</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>151</xmax>
			<xmin>94</xmin>
			<ymax>244</ymax>
			<ymin>166</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>204</xmax>
			<xmin>156</xmin>
			<ymax>210</ymax>
			<ymin>157</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Frontal</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>350</xmax>
			<xmin>299</xmin>
			<ymax>216</ymax>
			<ymin>157</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>bottle</name>
		<bndbox>
			<xmax>225</xmax>
			<xmin>215</xmin>
			<ymax>230</ymax>
			<ymin>196</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>bottle</name>
		<bndbox>
			<xmax>180</xmax>
			<xmin>170</xmin>
			<ymax>210</ymax>
			<ymin>184</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmax>179</xmax>
			<xmin>110</xmin>
			<ymax>244</ymax>
			<ymin>160</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Right</pose>
		<truncated>1</truncated>
	</object>
	<object>
		<name>chair</name>
		<bndbox>
			<xmax>87</xmax>
			<xmin>66</xmin>
			<ymax>270</ymax>
			<ymin>192</ymin>
		</bndbox>
		<difficult>0</difficult>
		<occluded>1</occluded>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
	</object>
	<segmented>0</segmented>
	<size>
		<depth>3</depth>
		<height>375</height>
		<width>500</width>
	</size>
	<source>
		<annotation>PASCAL VOC2009</annotation>
		<database>The VOC2009 Database</database>
"""
