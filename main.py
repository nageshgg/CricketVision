from utils import read_video, save_video
from trackers import Tracker
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #Readindg video
    video_frames = read_video('input_video/cricket.mp4')


    tracker = Tracker('model/best.pt')

    tracks  = tracker.get_object_tracks(video_frames, read_fram_stub=True, stub_path='stubs/track_stubs.pk1')

    tracker.add_position_to_tracks(tracks)
    # print(tracks['ball'][1]['position'])

    #Interpolate ball positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks['ball'])

    #  Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    #draw annotation
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    #save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print('Hello World')

if __name__ == '__main__':
    main()