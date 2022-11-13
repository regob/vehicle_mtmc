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


def dist(latlon1, latlon2) -> float:
    """Distance between two gps coordinates.

    Uses the Haversine formula (https://www.movable-type.co.uk/scripts/latlong.html)

    Parameters:
    ----------
    latlon1: array_like[2]
        Latitude-longitude of point 1.
    latlon2: array_like[2]
        Latitude-longitude of point 2.
    
    Returns:
    -------
    d: float
        Distance between the two points in meters.
    """
    R = 6371_000
    phi1 = latlon1[0] * math.pi / 180
    phi2 = latlon2[0] * math.pi / 180
    dphi = phi2 - phi1
    dlambda = (latlon2[1] - latlon1[1]) * math.pi / 180
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

def dist_planar(latlon1, latlon2):
    """Estimation of the distance between two points assuming planar earth."""
    R = 6371_000
    y1 = latlon1[0] * math.pi / 180
    y2 = latlon2[0] * math.pi / 180
    x1 = latlon1[1] * math.pi / 180
    x2 = latlon2[1] * math.pi / 180

    dy = abs(y1 - y2) * R
    r_avg = R * math.cos((y1 + y2) / 2)
    dx = abs(x1 - x2) * r_avg
    return math.sqrt(dx ** 2 + dy ** 2)
    
    
