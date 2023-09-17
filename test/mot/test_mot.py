import os
import tempfile
import pandas as pd

from config.defaults import get_cfg_defaults
from mot.run_tracker import run_mot, MOT_OUTPUT_NAME
from mot.tracklet_processing import load_tracklets
from tools import log
from tools import conversion

HIGHWAY_CONFIG = "examples/mot_highway.yaml"

def test_mot_highway():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, HIGHWAY_CONFIG))

    out_dir = tempfile.TemporaryDirectory()
    cfg.OUTPUT_DIR = out_dir.name
    cfg.MOT.SHOW = False
    cfg.MOT.VIDEO_OUTPUT = False
    cfg.DEBUG_RUN = True
    cfg.freeze()

    res = run_mot(cfg)
    assert res is not None

    # there should be no errors
    assert log.num_errors == 0

    # test csv output
    csv_file = os.path.join(out_dir.name, f"{MOT_OUTPUT_NAME}.csv")
    assert os.path.isfile(csv_file)
    df = pd.DataFrame(conversion.load_csv_format(csv_file))
    for col in ["frame", "track_id", "bbox_topleft_x", "bbox_topleft_y", "bbox_width", "bbox_height"]:
        assert col in df.columns

    # test txt (MOTChallenge output)
    txt_file = os.path.join(out_dir.name, f"{MOT_OUTPUT_NAME}.txt")
    assert os.path.isfile(txt_file)
    coldict = conversion.load_motchallenge_format(txt_file)
    df_motch = pd.DataFrame(coldict)
    for col in ["frame", "track_id", "bbox_topleft_x", "bbox_topleft_y", "bbox_width", "bbox_height"]:
        assert col in df_motch.columns

    assert len(df) == len(df_motch)
    assert (df["frame"] == df_motch["frame"]).all()

    # test pkl output
    pkl_file = os.path.join(out_dir.name, f"{MOT_OUTPUT_NAME}.pkl")
    assert os.path.isfile(pkl_file)
    tracks = load_tracklets(pkl_file)
    assert len(tracks) > 0

    # still no errors
    assert log.num_errors == 0
