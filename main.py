from utils import read_video, save_video
from trackers import Tracker, Pitch_Tracker
from ultralytics import YOLO
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from detect_ball_type import Ball_Type
import cv2

def main():
    #Readindg video
    video_frames = read_video('input_video/cricket.mp4')

    tracker = Tracker('model/best.pt')
    tracks  = tracker.get_object_tracks(video_frames, read_fram_stub=True, stub_path='stubs/track_stubs.pk1')

    pitchTracker = Pitch_Tracker('pitch-detection-model/best.pt')
    pitchTrackes = pitchTracker.get_object_tracks(video_frames, read_fram_stub=True, stub_path='stubs/pitch_track_stubs.pk1')

    tracker.add_position_to_tracks(tracks)

    #Interpolate ball positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks['ball'])

    #  Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #check for ball type:
    # ball_type = Ball_Type()
    # ball_type.detect_ball_type(tracks)




    #draw annotation
    output_video_frames, ball_dict = tracker.draw_annotations(video_frames, tracks)
    
    output_video_frames = pitchTracker.draw_annotations(video_frames, pitchTrackes, tracks)


    ## Draw Speed and Distance
    output_video_frames= speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print('Hello World')

if __name__ == '__main__':
    main()