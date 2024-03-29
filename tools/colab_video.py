import io
from IPython.display import HTML
from base64 import b64encode
import subprocess
import time
import os


def play_video(path, duration="total", offset=0, width=600):
    """ Returns HTML for playing an mp4 video for a duration

    Parameters
    ----------
    path : str
        The path to the video file
    duration: str or int
        Length of video chunk to be played in seconds.
    offset: int
        Beginning timestamp of playback in seconds.
    width: int
        Width of the HTML video frame.

    Returns
    -------
    HTML
    """

    if not os.path.exists(path):
        raise ValueError("Wrong path provided")

    if duration != "total" or offset > 0:
        tmp_file = "/tmp/{}.mp4".format(str(time.time()).split('.')[1])
        dur = [] if duration == "total" else ["-t", str(duration)]
        off = [] if offset == 0 else ["-ss", str(offset)]
        params = ["ffmpeg", "-i", path] + dur + off + [tmp_file]
        sproc = subprocess.run(params)
        path = tmp_file

    with io.open(path, 'r+b') as f:
        mp4 = f.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=%s controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % (str(width), data_url))
