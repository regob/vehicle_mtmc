import re
import glob
import os
from PIL import Image
import numpy as np


class ZoneMatcher:
    def __init__(self, zone_folder, valid_zone_paths):
        zone_files = glob.glob(os.path.join(zone_folder, "zone*"))
        self.masks = {}
        for path in zone_files:
            folder, filename = os.path.split(path)
            zone_id = int(filename[4:].split(".")[0])
            if zone_id == 0:
                raise ValueError(
                    "Zone 0 is reserved for the complement of all zones!")
            img = Image.open(path)
            mask = (np.array(img) / 180).astype(np.uint8)
            self.masks[zone_id] = mask

        self.valid_paths = [re.compile(path_regex)
                            for path_regex in valid_zone_paths]

    def find_zone_for_point(self, x, y):
        """ Find the zone in which the point (x, y) is in, if none then zone 0 is assumed. """

        for zone_id, mask in self.masks.items():
            if mask[y, x, 0] > 0:
                return zone_id
        return 0

    def find_zone_path(self, zone_list):
        """ Check which predefined zone path a zone list corresponds to. Path -1 means no match. """

        if type(zone_list) == list:
            zone_list = ",".join(map(str, zone_list))

        for idx, path_regex in enumerate(self.valid_paths):
            if path_regex.match(zone_list):
                return idx
        return -1

    def is_valid_path(self, zone_list):
        """ Check whether a zone list is valid for any path (i.e not in Path -1) """
        return self.find_zone_path(zone_list) >= 0
