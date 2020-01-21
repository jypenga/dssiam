import pdb
import argparse, cv2, os
import numpy as np
import sys
from imutils.video import FPS
import json

from tracker import TrackerSiamFC
from tracker import DSSiam, SiamFC

# constants
BRIGHTGREEN = [102, 255, 0]
RED = [0, 0, 255]
YELLOW = [0, 255, 255]
np.set_printoptions(precision=6, suppress=True)

OUTPUT_WIDTH = 740
OUTPUT_HEIGHT = 555
PADDING = 2

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False

LIMIT = 99999999
def xyxy_to_xywh(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    x1 = bboxes[0,...]
    y1 = bboxes[1,...]
    x2 = bboxes[2,...]
    y2 = bboxes[3,...]
    bboxesOut[0,...] = (x1 + x2) / 2.0
    bboxesOut[1,...] = (y1 + y2) / 2.0
    bboxesOut[2,...] = x2 - x1
    bboxesOut[3,...] = y2 - y1
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    return bboxesOut

def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize, boxToDraw_xywh
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0, 2]] = np.sort(boxToDraw[[0, 2]])
    boxToDraw[[1, 3]] = np.sort(boxToDraw[[1, 3]])
    boxToDraw_xywh = xyxy_to_xywh(boxToDraw)

def show_webcam(tracker, mirror=False, viz=False):
    global initialize

    vs = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0)

    outputBoxToDraw = None
    bbox = None
    fps = None

    # loop over video stream ims
    while True:
        _, im = vs.read()

        if mirror:
            im = cv2.flip(im, 1)

        if mousedown:
            (x1, y1, x2, y2) = [int(l) for l in boxToDraw]
            cv2.rectangle(im, (x1, y1), (x2, y2),
                          BRIGHTGREEN, PADDING)

        elif mouseupdown:
            if initialize:
                tracker.init(im, xyxy_to_xywh([x1, y1, x2, y2]))
                initialize = False
                fps = FPS().start()
            else:
                box = np.round(tracker.update(im)).astype(int)

                fps.update()
                fps.stop()

                # Display the image
                info = [
                    ("FPS:", f"{fps.fps():.2f}"),
                ]

                cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 255), 3)

                # for (i, (k, v)) in enumerate(info):
                #     text = "{}: {}".format(k, v)
                #     cv2.putText(im, text, (10, OUTPUT_HEIGHT - ((i * 20) + 20)),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Webcam", im)

        # check for escape key
        key = cv2.waitKey(1)
        if key==27 or key==1048603:
            break

    # release the pointer
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = TrackerSiamFC(backbone=DSSiam(n=1), netpath='models/dssiam_n2_e50.pth')

    print("[INFO] Starting video stream.")
    show_webcam(tracker, mirror=True)
