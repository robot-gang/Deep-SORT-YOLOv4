#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

#our imports
from datetime import datetime
import config
import sys

warnings.filterwarnings('ignore')


def main(yolo, local_params):

    camera_capture = local_params['camera_capture']
    file_path = local_params['input_file_path']

    # Definition of the parameters
    max_cosine_distance = local_params['max_cosine_distance']
    nn_budget = local_params['nn_budget']
    nms_max_overlap = local_params['nms_max_overlap']

    # Deep SORT
    model_filename = local_params['model_filename']
    encoder = gdet.create_box_encoder(model_filename, batch_size=local_params['encoder_batch_size'])

    metric = nn_matching.NearestNeighborDistanceMetric(local_params['distance_metric'], max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = local_params['show_detections']
    writeVideo_flag = local_params['write_video_flag']
    asyncVideo_flag = local_params['async_video_flag']

    if camera_capture:
        #using camera capture
        video_capture = cv2.VideoCapture(0)
    else:
        if asyncVideo_flag:
            video_capture = VideoCaptureAsync(file_path)
        else:
            video_capture = cv2.VideoCapture(file_path)

        if asyncVideo_flag:
            video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(local_params['output_file'] + '_' + str(datetime.now()) + '.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for det in detections:
            bbox = det.to_tlbr()
            if show_detections and len(classes) > 0:
                det_cls = det.cls
                score = "%.2f" % (det.confidence * 100) + "%"
                cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 1)
            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0),
                            1)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    local_params = config.load_file()
    main(YOLO(), local_params)
