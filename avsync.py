import numpy as np
import cv2
from tqdm import tqdm
import os


def combine_vid_and_audio(wavfile, vidfile, syncfile, outvidfile, vid_fps=30, audio_fs=44100, cam_delay=0.0):

    tmp_vidfile = './data/tmp.mp4'

    T = np.genfromtxt(syncfile, delimiter=',', skip_header=1)

    T[:,1] = T[:,1]/audio_fs
    t = np.arange(0, T[-10,1], 1/vid_fps)
    n_frame = t.shape[0]

    vr = cv2.VideoCapture(vidfile)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter(tmp_vidfile, fmt, vid_fps, (int(vr.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT))), isColor=True)

    crnt_ts = -10
    cnt = 0
    for i_frame in tqdm(range(n_frame)):
        while crnt_ts+cam_delay < t[i_frame]:
            ret, frame = vr.read()
            crnt_ts = T[cnt,1]
            cnt += 1
        vw.write(frame)

    vw.release()
    vr.release()

    os.system('ffmpeg -y -i "' + tmp_vidfile + '" -i "' + wavfile + '" -c:v copy -c:a aac "' + outvidfile + '"')


if __name__ == '__main__':

    wavfile = './recording.wav'
    vidfile = './vid.mp4'
    syncfile = './sync.csv'
    outvidfile = './video_audio.mp4'
    vid_fps = 30
    audio_fs = 44100
    cam_delay = 0.0

    combine_vid_and_audio(wavfile, vidfile, syncfile, outvidfile, vid_fps, audio_fs, cam_delay)