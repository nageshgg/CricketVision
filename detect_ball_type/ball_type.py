import cv2 
import numpy as np


class Ball_Type():

    def __init__(self):
        self.pitching_zone_y  = 700
        self.yorker_zone = 20

    def detect_ball_type(self, tracks):

        ball_positions = []

        for object, object_tracks in tracks.items():
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames):
                for track_id,_ in object_tracks[frame_num].items():
                    if object_tracks[frame_num][track_id]['position'] is not None:
                        ball_positions.append(object_tracks[frame_num][track_id]['position'])

        print(ball_positions)

        if len(ball_positions) > 1:
            for i in range(1, len(ball_positions)):
                if ball_positions[i][1] > self.pitching_zone_y and \
                abs(ball_positions[i][1] - ball_positions[i-1][1]) < self.yorker_zone:
                    print("Yorker detected at position:", ball_positions[i])