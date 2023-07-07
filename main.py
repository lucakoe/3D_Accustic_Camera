import os
import cv2
import numpy as np
import pyaudio
import wave
import csv
import time
import threading
import avsync


# Global variable for current audio frame
current_audio_frame = 0
start_time = 0

# General settings
duration = 5  # in seconds
audio_recording_out_path = './recording.wav'
video_recording_out_path = './recording.mp4'
syncfile_path = './audio_video_timestamps.csv'
output_path = './video_audio.mp4'

# Audio settings
num_channels = 16
sample_rate = 48000  # carefull use the same as the audio device in the system settings
chunk = 1024

# Video settings
width = 640
height = 480
fps = 25
cam_delay = 0.0


def record_audio(stream, audio_frames, num_audio_frames):
    global current_audio_frame, sample_rate, start_time
    start_time = time.time()
    for i in range(num_audio_frames):
        # Calculate the expected timestamp of the current audio frame
        expected_audio_time = current_audio_frame * (duration / num_audio_frames)

        # Wait until the expected timestamp is reached
        while (time.time() - start_time) < expected_audio_time:
            time.sleep(0.001)

        # Capture audio frame
        data = stream.read(chunk)
        audio_frames.append(data)
        current_audio_frame += 1


def record_video(cap, video_out, frames, num_video_frames, csv_writer):
    global current_audio_frame, sample_rate, width, height, fps
    for i in range(num_video_frames):
        # Capture video frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            video_out.write(frame)
            frames.append(frame)

            audio_position_file = current_audio_frame * chunk
            # Save the timestamps to the CSV file
            csv_writer.writerow([time.time() - start_time, i + 1, current_audio_frame, audio_position_file])


def record():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new audio stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=num_channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    # Initialize video recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(video_recording_out_path, fourcc, fps, (width, height))

    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW,)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    # Initialize CSV writer
    with open(syncfile_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Time (s)', 'Video Frame Number', 'Audio Frame Number', 'Audio Position in file'])

        # Start recording audio in a separate thread
        audio_frames = []
        num_audio_frames = int(duration * sample_rate / chunk)
        audio_thread = threading.Thread(target=record_audio, args=(stream, audio_frames, num_audio_frames))
        audio_thread.start()

        # Start recording video in a separate thread
        frames = []
        num_video_frames = int(duration * fps)
        video_thread = threading.Thread(target=record_video,
                                        args=(cap, video_out, frames, num_video_frames, csv_writer))
        video_thread.start()

        # Wait for both threads to finish
        audio_thread.join()
        video_thread.join()

        # Stop recording and close the audio and video streams
        stream.stop_stream()
        stream.close()
        audio.terminate()
        video_out.release()
        cap.release()
        cv2.destroyAllWindows()

        # Save the recorded audio to a WAV file
        wave_file = wave.open(audio_recording_out_path, "wb")
        wave_file.setnchannels(num_channels)
        wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(b"".join(audio_frames))
        wave_file.close()

        # Check the length of the audio and video data arrays
        print("Length of audio data:", len(audio_frames))
        print("Length of video data:", len(frames))

        # Close the CSV file
        csv_file.close()


if __name__ == '__main__':
    record()
    avsync.combine_vid_and_audio(audio_recording_out_path, video_recording_out_path, syncfile_path, output_path, fps,
                                 sample_rate, cam_delay)
