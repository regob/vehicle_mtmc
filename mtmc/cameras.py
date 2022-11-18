

class CameraLayout:
    """A camera layout for multi camera tracking."""
    def __init__(self, camera_layout_path: str):
        self.offset, self.compatibility = [], []
        self.dtmin, self.dtmax = [], []
        self.fps = []
        self.scales = []
        self.n_cams = None

        f = open(camera_layout_path, "r")

        def numberline(_type):
            return list(map(_type, f.readline().strip().split()))
        
        line = f.readline()
        while line:
            if not line:
                break
            line = line.strip()
            if line.startswith("#") or not line:
                line = f.readline()
                continue

            if line == "fps":
                self.fps = numberline(float)
            elif line == "offset":
                self.offset = numberline(float)
            elif line == "scale":
                self.scales = numberline(float)
            elif line == "compatibility":
                line = numberline(int)
                self.compatibility.append(line)
                self.n_cams = len(line)
                for _ in range(self.n_cams - 1):
                    line = numberline(int)
                    self.compatibility.append(line)
            elif line in ("dtmin", "dtmax"):
                matrix = self.dtmin if line == "dtmin" else self.dtmax
                line = numberline(float)
                matrix.append(line)
                self.n_cams = len(line)
                for _ in range(self.n_cams - 1):
                    line = numberline(float)
                    matrix.append(line)
            else:
                raise ValueError(f"Error when parsing camera layout at line: '{line}'")
                
            line = f.readline()

        """some asserts to make sure the file was loaded correctly..."""
        for val in [self.n_cams, self.offset, self.fps, self.dtmax, self.dtmin, self.compatibility]:
            assert val, "Missing sections from camera layout file"
        assert len(self.dtmin) == self.n_cams
        assert len(self.dtmax) == self.n_cams
        assert len(self.compatibility) == self.n_cams
        assert len(self.offset) == self.n_cams
        assert len(self.fps) == self.n_cams
        for i in range(self.n_cams):
            assert len(self.dtmin[i]) == self.n_cams
            assert len(self.dtmax[i]) == self.n_cams
            assert len(self.compatibility[i]) == self.n_cams


        def to_bitmap(vals):
            bmp = 0
            for idx, val in enumerate(vals):
                if val:
                    bmp |= 1 << idx
            return bmp
        
        """self.compatibility converted to bitmaps (jth bit means compatibility with c_j)."""
        self._compatibility_bitmaps = [to_bitmap(vals) for vals in self.compatibility]

    @property
    def cam_compatibility_bitmaps(self):
        return self._compatibility_bitmaps
    
    def cam_compatibility_bitmap(self, cam_idx):
        return self._compatibility_bitmaps[cam_idx]

            
    
if __name__ == "__main__":
    # test a layout file
    path = "../config/mtmc_camera_layout.txt"
    cam = CameraLayout(path)
    print(f"n_cams: {cam.n_cams}")
    print(f"offset: {cam.offset}")
    print(f"fps: {cam.fps}")
    print(f"compatibility: {cam.compatibility}")
    print(f"dtmin: {cam.dtmin}")
    print(f"dtmax: {cam.dtmax}")

    
