# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import collections, operator
import time
import json
import cv2
import csv
import numpy

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
previousImage = None
absDiffHistory = collections.deque(maxlen=20)
max_idx = 10
current_idx = 0
template_message = "Press l for landfill, c for compost, r for recycle"
nextMessage = template_message

def file_get_content(filename):
    fp = open(filename, 'r')
    content = fp.read()
    fp.close()
    return content

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # Do a bunch of processing 

    SAD = None

    if previousImage is not None:
        SAD = sum(cv2.sumElems(cv2.absdiff(previousImage, image)))

    # if we have enough history, check for changes
    if len(absDiffHistory) > 18:
        prevLast = absDiffHistory[-1]
        absDiffHistory.append(SAD)
        stddev = numpy.std(absDiffHistory)
        mean = sum(absDiffHistory)/len(absDiffHistory)
        threshold = prevLast * -1.0 + mean * 2.0 + stddev * 2.0
        if threshold < SAD:
            file_name = '/var/tmp/{0}.jpg'.format(current_idx % max_idx)
            cv2.imwrite(file_name, image)
            current_idx += 1
            print("Over threshold and not blur! ", file_name)

    elif SAD is not None:
        # just append and do nothing
        absDiffHistory.append(SAD)

    previousImage = image

    # show the frame
    cv2.putText(image, nextMessage, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 255), 3)
    try:
        with open('/var/tmp/topics.csv') as infile:
            reader = csv.reader(infile, delimiter='\t')
            topics = {row[0] : row[1] for row in reader}
            topics = {k:v for k, v in topics.iteritems() if float(v) > 0.1}
            sorted_topics = sorted(topics.items(), key=operator.itemgetter(1), reverse=True)[:3]

        idx = 0
        for k, v in sorted_topics:
            cv2.putText(image, k, (100, 60 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 255), 3)
            idx += 1
    except Exception as e:
        print('show exception', e)
    cv2.imshow("Frame", image)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("c"):
        nextMessage = "It is compost!"
        time.sleep(0.5)
    elif key == ord("r"):
        nextMessage = "It is recycle!"
        time.sleep(0.5)
    elif key == ord("l"):
        nextMessage = "It is landfill!"
        time.sleep(0.5)
    else:
        nextMesage = template_message

