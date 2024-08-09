from utils import read_video, save_video
from trackers import Tracker

def main():
    #Readindg video
    video_frames = read_video('input_video/cricket.mp4')

    tracker = Tracker('model/best.pt')
    tracks  = tracker.get_object_tracks(video_frames, read_fram_stub=True, stub_path='stubs/track_stubs.pk1')

    #draw annotation
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print('Hello World')

if __name__ == '__main__':
    main()