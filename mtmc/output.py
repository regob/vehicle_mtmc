from typing import List
import argparse
import pickle

from mtmc.multicam_tracklet import MulticamTracklet, get_tracks_by_cams
from mot.video_output import annotate_video_with_tracklets
from mot.tracklet_processing import save_tracklets, save_tracklets_csv


def load_mtmc_tracklets(path: str):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res


def save_mtmc_tracklets(multicam_tracks: List[MulticamTracklet], path: str):
    with open(path, "wb") as f:
        pickle.dump(multicam_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_tracklets_per_cam(multicam_tracks: List[MulticamTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets(tracks, path)


def save_tracklets_csv_per_cam(multicam_tracks: List[MulticamTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets_csv(tracks, path)


def annotate_video_mtmc(video_in, video_out, multicam_tracks, cam_idx, **kwargs):
    tracks = get_tracks_by_cams(multicam_tracks)[cam_idx]
    annotate_video_with_tracklets(video_in, video_out, tracks, **kwargs)

                                  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="annotate a video from multicam tracks")
    parser.add_argument("--cam_idx", type=int, required=True)
    parser.add_argument("--tracklets", required=True,
                        help="multicam tracklets pickle")
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--video_out", required=True)
    args = parser.parse_args()

    tracks = load_mtmc_tracklets(args.tracklets)
    annotate_video_mtmc(args.video_in, args.video_out, tracks, args.cam_idx)
