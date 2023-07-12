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


import cv2

import harvester

# Create a new Harvester instance
harvester_instance = harvester.Harvester()

# Use the Harvester instance to scrape a website
data = harvester_instance.scrape('http://example.com')

# Print the scraped data
print(data)
# Create a new Harvester object
harvester = Harvester()

# Set the camera's IP address and other parameters
ip_address = "192.168.0.100"
packet_size = 1500
timeout = 1000

# Connect to the camera
harvester.add_cti_file("C:/Program Files/JAI/SDK/bin/x64/JaiGigE_V_1_2.cti")
camera = harvester.create_image_acquirer(ip_address, packet_size, timeout)

# Start image acquisition
camera.start_acquisition()

# Acquire and display images
while True:
    # Retrieve the next image from the camera
    image = camera.get_image()

    # Convert the image to a numpy array
    frame = cv2.cvtColor(image.as_opencv_image(), cv2.COLOR_BGR2RGB)

    # Display the image
    cv2.imshow("Camera", frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop image acquisition and release resources
camera.stop_acquisition()
cv2.destroyAllWindows()