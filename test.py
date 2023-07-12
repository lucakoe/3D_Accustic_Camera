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

import clr
import System.Reflection
import sys

# Add the Python.NET library to the system path
sys.path.append(r"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages")

# Import the necessary classes and modules from the .NET Framework and the ICImagingControl namespace
from System.Reflection import Assembly
from TIS.Imaging.ICImagingControl import ImagingControl, DeviceEnumeration

def discover_cameras():
    # Load the ICImagingControl.dll file
    ic_dll = Assembly.LoadFrom(r"C:\Program Files\The Imaging Source Europe GmbH\IC Imaging Control 3.5\ClassLib\ICImagingControl.dll")

    # Enumerate all available devices
    devices = DeviceEnumeration.EnumerateDevices()

    # Print information about each discovered camera
    print(f"Found {len(devices)} camera(s):")
    for i, device in enumerate(devices):
        print(f"Camera {i+1}:")
        print(f"\tDevice Name: {device.DeviceName}")
        print(f"\tSerial Number: {device.SerialNumber}")
        print(f"\tDisplay Name: {device.DisplayName}")
        print(f"\tDevice Path: {device.DevicePath}")

if __name__ == "__main__":
    discover_cameras()