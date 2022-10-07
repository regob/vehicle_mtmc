import math
import numpy as np

class Projector:
    """Projects 2d points to 3d space defined by a homography matrix."""
    
    def __init__(self, camera_matrix_file: str):

        # homography matrix: 3d -> 2d
        self.homography = np.zeros((3, 3))
        with open(camera_matrix_file, "r") as f:
            line = f.readline()
            PRE = "Homography matrix:"
            if line.startswith(PRE):
                line = line[len(PRE):]
            rows = line.split(";")
            for i, line in enumerate(rows):
                cols = line.strip().split()
                assert len(cols) == 3
                for j, x in enumerate(cols):
                    self.homography[i, j] = float(x)
        self.inv_homography = np.linalg.inv(self.homography)
        
    def project3d(self, x, y):
        p =  np.matmul(self.inv_homography, np.array([x, y, 1]))
        p /= p[2]
        return p[:2]


def dist(latlon1, latlon2):
    """Distance between two gps coordinates."""
    R = 6371_000
    phi1 = latlon1[0] * math.pi / 180
    phi2 = latlon2[0] * math.pi / 180
    dphi = phi2 - phi1
    dlambda = (latlon2[1] - latlon1[1]) * math.pi / 180
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d
    
def average_speed(coords: list, total_frames: int, frame_rate: float):
    if total_frames == 0:
        return 0.0
    dists = [dist(coords[i], coords[i+1]) for i in range(len(coords)-1)]
    return sum(dists) * (frame_rate / total_frames) * 3.6

def to_point(row):
    x = round(row["bbox_topleft_x"] + row["bbox_width"] / 2)
    y = row["bbox_topleft_y"] + row["bbox_height"]
    return x, y

if __name__ == "__main__":
    proj = Projector("../datasets/cityflow_track3/validation/S02/c006/calibration.txt")
    import pandas as pd
    df = pd.read_csv("../output/cityflow_s02_city/0_vdo/vdo_mtmc.csv")
    for track_id, ddf in df.groupby("track_id"):
        p = []
        first = ddf.iloc[0]["frame"]
        last = ddf.iloc[len(ddf)-1]["frame"]
        for i in range(len(ddf)):
            x1, y1 = to_point(ddf.iloc[i])
            p1 = proj.project3d(x1, y1)
            p.append(p1)
        print(f"Track {track_id} speed: {average_speed(p, last-first, 10)} km/h")
