import os
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import h5py
import numpy as np
import scipy.io.wavfile
import cv2


def generateCalibrationTone():
    # Create an empty AudioSegment
    result = AudioSegment.silent(duration=0)  # Loop over 0-14
    tone = AudioSegment.silent(duration=0)  # Loop over 0-14
    for n in range(10, 100):  # Generate a sine tone with frequency 200 * n
        gen = Sine(100 * n)  # AudioSegment with duration 200ms, gain -3
        sine = gen.to_audio_segment(duration=1)  # Fade in / out
        sine = sine.fade_in(10).fade_out(10)  # Append the sine to our result
        tone += sine  # Play the result

    for m in range(180):
        result += tone
        result += AudioSegment.silent(duration=1000)  # Loop over 0-14

    result.export("ascending_tone.wav", format="wav")


def generateMicPosFile(mic_array_position, mic_array_arrangement=[
    [0.0, 0.0, 0.0],  # Mic 14
    [42.0 * 1, 0.0, 0.0],  # Mic 12
    [0.0, 42.0 * 1, 0.0],  # Mic 3
    [42.0 * 1, 42.0 * 1, 0.0]]  # Mic 5
                       ):
    # Create datasets
    result = np.add(np.array(mic_array_arrangement), np.array(mic_array_position))

    result = result / 1000

    with h5py.File("./data/micpos.h5", mode='w') as f:
        f.create_dataset('/result/micpos', data=result)


def checkMicPosFile(path):
    with h5py.File(path, 'r') as f:
        print(f.keys())
        print(f['result/micpos'][()])


def wav2dat(data_dir):
    A = scipy.io.wavfile.read(data_dir + '/cut_audio.wav')
    max_int16 = 32767
    max_int32 = 2147483647
    X = A[1]

    with open(data_dir + '/snd.dat', 'wb') as f:
        f.write(X.tobytes())


def create_paramfile(data_dir, image_width=640, image_height=768, daq_fs=192000, daq_n_ch=4, camera_fov_y=1.047198, #TODO in rad??? Logi C210 --> 60 deg
                     camera_height=2.0):
    pressure_calib = np.array([1, 1, 1, 1], dtype=float)
    mic0pos = np.array([0.015, -0.03, 0.07], dtype=float)

    speedOfSound = 343.0

    fy = image_height / (2 * np.tan(camera_fov_y / 2))

    fx = fy

    ppx = image_width / 2
    ppy = image_height / 2

    r = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    t = np.array([0, 0, 0], dtype=float)
    coeff = np.array([0, 0, 0, 0, 0], dtype=float)

    with h5py.File(data_dir + '/param.h5', mode='w') as f:
        f.create_dataset('/camera_param/camera_height', data=camera_height)
        f.create_dataset('/daq_param/fs', data=daq_fs)
        f.create_dataset('/daq_param/n_ch', data=daq_n_ch)
        f.create_dataset('/misc/speedOfSound', data=speedOfSound)
        f.create_dataset('/misc/pressure_calib', data=pressure_calib)
        f.create_dataset('/misc/mic0pos', data=mic0pos)
        f.create_dataset('/camera_param/color_intrin/coeffs', data=coeff)
        f.create_dataset('/camera_param/color_intrin/fx', data=fx)
        f.create_dataset('/camera_param/color_intrin/fy', data=fy)
        f.create_dataset('/camera_param/color_intrin/width', data=image_width)
        f.create_dataset('/camera_param/color_intrin/height', data=image_height)
        f.create_dataset('/camera_param/color_intrin/ppx', data=ppx)
        f.create_dataset('/camera_param/color_intrin/ppy', data=ppy)
        f.create_dataset('/camera_param/depth_to_color_extrin/rotation', data=r)
        f.create_dataset('/camera_param/depth_to_color_extrin/translation', data=t)


if __name__ == '__main__':
    #generateMicPosFile([13, -35, 8])
    #checkMicPosFile('usvcam_main/test_data/micpos.h5')
    # checkMicPosFile('usvcam_main/test_data/micpos_custom.h5')
    #checkMicPosFile(('data/micpos.h5'))
    #wav2dat("./data/2023-07-11-15-14-36")
    create_paramfile("./data/2023-07-12-10-27-46",640,480,48000,4)


