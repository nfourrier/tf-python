import os
import numpy as np
import cv2
import datetime as dt

# synset = [l.strip() for l in open('./synset.txt').readlines()]
#


utc = dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
prefix = 'test'
sequence = utc

folder = '{}_{}'.format(prefix,sequence)
folder = 'test'

cap = cv2.VideoCapture(-1)

ramp_frames = 30

if not os.path.exists(folder):
    try:
        os.makedirs(folder)
        print("Folder %s has been created" % (str(folder)))
    except Exception as e:
        err = "Path does not exist and an error was encountered when attempting to create it"
count = 0
while(True):
    count = count+1
    # Capture frame-by-frame
    ret, frame = cap.read()


    if(count > ramp_frames):
        print(count)
        filename = os.path.join(folder,'{}_{}_{}.JPEG'.format(prefix,sequence,count))
        cv2.imwrite(filename, frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






