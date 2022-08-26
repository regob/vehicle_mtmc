import argparse
import pickle

from mot.video_output import annotate_video_with_tracklets
from mtmc.multicam_tracklet import MulticamTracklet

def load_tracklets(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res

def annotate_video(video_in, video_out, multicam_tracks, cam_idx):
    tracks = []
    for mtrack in multicam_tracks:
        for track in mtrack.tracks:
            if track.cam == cam_idx:
                tracks.append(track)
    annotate_video_with_tracklets(video_in, video_out, tracks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="annotate a video from multicam tracks")
    parser.add_argument("--cam_idx", type=int, required=True)
    parser.add_argument("--tracklets", required=True, help="multicam tracklets pickle")
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--video_out", required=True)
    args = parser.parse_args()

    tracks = load_tracklets(args.tracklets)
    annotate_video(args.video_in, args.video_out, tracks, args.cam_idx)
    
