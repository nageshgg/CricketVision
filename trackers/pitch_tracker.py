from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, is_yorker

class Pitch_Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
 
    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.20)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_fram_stub= False, stub_path=None):

        if read_fram_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)


        tracks = {
            "pitch" : []
        }

        for frame_num, detection  in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks["pitch"].append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                print('achecking BBOx', bbox)
                print(cls_id, cls_names_inv)

                if cls_id == cls_names_inv['pitch']:
                    print('inside')
                    tracks["pitch"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_line(self,frame,bbox,ball_position,color):
        # y= int(bbox[1])
        x1, y1, x2, y2  = map(int, bbox)
        image_height, image_width = frame.shape[:2]
        # x, y = get_center_of_bbox(ball_bbox)
        is_yorker_ball = is_yorker( ball_position ,x1 ,y1 , x2+20 ,y1+20)
        text = f'Is Yorker: {is_yorker_ball}'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = image_width - text_width - 10  # 10 pixels from the right edge
        y = image_height - 10  # 10 pixels from the bottom edge
        
        cv2.line(frame, (x1,y1), (x2, y1),color, thickness=2)
        cv2.line(frame, (x1,y2), (x2, y2), color, thickness=2)
        cv2.rectangle(frame, (x1,y1), ((x2+20), (y1+20)), color, thickness=2)
        
        # cv2.putText(frame,f'Is Yorker: {is_yorker_ball}',(x, y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    def draw_annotations(self,video_frames, tracks, ball_track):
        output_video_frames= []
        is_yorker_ball = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            pitch_dict = tracks["pitch"][frame_num]
            ball_dict = ball_track["ball"][frame_num]
            
            # print(pitch_dict)
            for track_id, pitch in pitch_dict.items():
                print('pitch', pitch_dict)
                if pitch['bbox'] is not None:
                    print('ball', ball_dict)
                    ball = ball_dict.get(track_id)
                    if ball and ball['bbox'] is not None:
                        print('kjashndfkj')
                        frame = self.draw_line(frame, pitch["bbox"], ball['position'], (0, 255, 0))

            output_video_frames.append(frame)
            
        return output_video_frames 