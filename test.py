import cv2
import numpy as np
import pyaudio
import wave
import csv
import time
import threading
import avsync
import os
import datetime
import calibration
import scipy.io.wavfile
import usvcam.analysis

# def cut_wav_channels(input_file, channels_to_keep, output_file):
#     with wave.open(input_file, 'rb') as wav:
#         channels = wav.getnchannels()
#         sample_rate = wav.getframerate()
#         wav_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
#         wav_data = np.reshape(wav_data, (-1, channels))
#         wav_data = wav_data[:, channels_to_keep]
#         with wave.open(output_file, 'wb') as new_wav:
#             new_wav.setnchannels(wav_data.shape[1])
#             new_wav.setsampwidth(wav.getsampwidth())
#             new_wav.setframerate(sample_rate)
#             new_wav.writeframes(wav_data.tobytes())
#     return output_file
# def wav2dat(data_dir):
#     A = scipy.io.wavfile.read(data_dir + '/cut_audio.wav')
#     max_int16 = 32767
#     max_int32 = 2147483647
#     X = A[1]
#
#     with open(data_dir + '/snd.dat', 'wb') as f:
#         f.write(X.tobytes())
#
# data_dir = os.path.join("data", "2023-07-11-16-40-28")
#
# audio_recording_out_filename = "audio.wav"
# cut_wav_channels(os.path.join(data_dir, audio_recording_out_filename), [3 - 1, 5 - 1, 12 - 1, 14 - 1],
#                  os.path.join(data_dir, 'cut_' + audio_recording_out_filename))
#
# wav2dat(data_dir)
# usvcam.analysis.dat2wav(data_dir, 3)


import PyCapture2

bus = PyCapture2.BusManager()
num_cams = bus.getNumOfCameras()

print("Number of cameras detected: ", num_cams)

if num_cams > 0:
    # Get the camera GUID
    cam = PyCapture2.GigECamera()
    cam.connect(bus.getCameraFromIndex(0))
    cam_info = cam.getCameraInfo()
    print("Camera model: ", cam_info.modelName)
    cam.disconnect()






