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
import usvcam_main.usvcam.analysis as analysis

import calibration

# Global variable for current audio frame
current_audio_frame = 0
start_time = 0

# General settings
audio_recording_out_filename = 'audio.wav'
video_recording_out_filename = 'vid.mp4'
syncfile_filename = 'sync.csv'
output_filename = 'video_audio.mp4'
parameter_filename = 'param.h5'
calib_path = './data/micpos.h5'
mic_array_amount = 2  # number of microphones
mic_array_position = [[-0.063, 0.032, 0.005], [0, 0, 0]]  # relative to camera
mic_array_layout = [8, 9, 6, 7, 10, 11, 4, 5, 12, 13, 2, 3, 14, 15, 0,
                    1]  # mic channels arranged from left top to right bottom

# Audio settings
num_channels = 16
sample_rate = 48000  # carefull use the same as the audio device in the system settings
chunk = 1024

# Video settings
width = 640
height = 480
fps = 25
cam_delay = 0.0


def record_audio(streams, audio_frames, stop_event):
    global current_audio_frame, sample_rate, start_time
    while not stop_event.is_set():
        # Calculate the expected timestamp of the current audio frame
        expected_audio_time = current_audio_frame * (chunk / sample_rate)

        # Wait until the expected timestamp is reached
        while (time.time() - start_time) < expected_audio_time and not stop_event.is_set():
            time.sleep(0.001)

        if stop_event.is_set():
            break

        # Capture audio frames from each stream
        data = []
        for stream in streams:
            data.append(stream.read(chunk))
        audio_frames.append(data)
        current_audio_frame += 1


def record_video(cap, video_out, frames, csv_writer, stop_event):
    global current_audio_frame, sample_rate, width, height, fps
    while not stop_event.is_set():
        # Capture video frame
        ret, frame = cap.read()
        if not ret or stop_event.is_set():
            break

        frame = cv2.resize(frame, (width, height))
        video_out.write(frame)
        frames.append(frame)

        audio_position_file = current_audio_frame * chunk
        # Save the timestamps to the CSV file
        csv_writer.writerow([time.time() - start_time, audio_position_file])

        # Display live preview
        cv2.imshow('Preview', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit preview
            break

    cv2.destroyAllWindows()


def record(output_path):
    print("Press Enter to start recording")
    input()

    print("Start recording\n")
    # Set the paths for the output files
    audio_recording_out_path = os.path.join(data_dir, audio_recording_out_filename)
    video_recording_out_path = os.path.join(data_dir, video_recording_out_filename)
    syncfile_path = os.path.join(data_dir, syncfile_filename)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open audio streams for recording from each microphone
    streams = []
    for i in range(mic_array_amount):
        stream = audio.open(format=pyaudio.paInt16,
                            channels=num_channels,
                            rate=sample_rate,
                            input=True,
                            input_device_index=i,  # use the correct microphone device index here
                            frames_per_buffer=chunk)
        streams.append(stream)

    # Initialize video recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(video_recording_out_path, fourcc, fps, (width, height))

    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW, )
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize CSV writer
    with open(syncfile_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(['Time (s)', 'Video Frame Number', 'Audio Frame Number', 'Audio Position in file'])

        # Start recording audio in a separate thread
        audio_frames = []
        stop_event = threading.Event()
        audio_thread = threading.Thread(target=record_audio, args=(streams, audio_frames, stop_event))
        audioframes = []
        audio_thread.start()

        # Start recording video in a separate thread
        frames = []
        record_event = threading.Event()
        video_thread = threading.Thread(target=record_video, args=(cap, video_out, frames, csv_writer, record_event))
        video_thread.start()

        # Wait for the user to stop recording
        input("Press Enter to stop recording\n")

        # Stop recording and join threads
        stop_event.set()
        record_event.set()
        audio_thread.join()
        video_thread.join()

        # Close audio streams and PyAudio
        for stream in streams:
            stream.stop_stream()
            stream.close()
        audio.terminate()

        # Release video capture and writer
        cap.release()
        video_out.release()

        # Save the recorded audio to WAV files for each microphone
        for i in range(mic_array_amount):
            if i == 0:
                audio_recording_out_path_i = audio_recording_out_path
            else:
                audio_recording_out_path_i = f"{audio_recording_out_path[:-4]}_{i}.wav"

            wave_file = wave.open(audio_recording_out_path_i, "wb")
            wave_file.setnchannels(num_channels)
            wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(sample_rate)
            audio_frames_i = [frame[i] for frame in audio_frames]  # extract audio frames from the i-th microphone
            wave_file.writeframes(b"".join(audio_frames_i))
            wave_file.close()

        # Check the length of the audio and video data arrays
        print("Length of audio data:", len(audio_frames))
        print("Length of video data:", len(frames))

        # Close the CSV file
        csv_file.close()

        print("Recording saved\n")


def cut_wav_channels(input_file, channels_to_keep, output_file):
    with wave.open(input_file, 'rb') as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        wav_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
        wav_data = np.reshape(wav_data, (-1, channels))
        wav_data = wav_data[:, channels_to_keep]
        with wave.open(output_file, 'wb') as new_wav:
            new_wav.setnchannels(wav_data.shape[1])
            new_wav.setsampwidth(wav.getsampwidth())
            new_wav.setframerate(sample_rate)
            new_wav.writeframes(wav_data.tobytes())
    return output_file


def rearrange_wav_channels(input_file, channel_order, output_file):
    with wave.open(input_file, 'rb') as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        wav_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
        wav_data = np.reshape(wav_data, (-1, channels))
        wav_data = wav_data[:, channel_order]
        with wave.open(output_file, 'wb') as new_wav:
            new_wav.setnchannels(wav_data.shape[1])
            new_wav.setsampwidth(wav.getsampwidth())
            new_wav.setframerate(sample_rate)
            new_wav.writeframes(wav_data.tobytes())
    return output_file


if __name__ == '__main__':
    # # Get the current date and time
    # now = datetime.datetime.now()
    # timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    #
    # # Create a new directory for this recording
    # data_dir = os.path.join("data", timestamp)
    # os.makedirs(data_dir)
    # record(data_dir)
    #
    # rearrange_wav_channels(os.path.join(data_dir, audio_recording_out_filename), mic_array_layout,
    #                        os.path.join(data_dir,
    #                                      audio_recording_out_filename))
    #
    # avsync.combine_vid_and_audio(os.path.join(data_dir,  audio_recording_out_filename),
    #                              os.path.join(data_dir, video_recording_out_filename),
    #                              os.path.join(data_dir, syncfile_filename),
    #                              os.path.join(data_dir, output_filename), fps, sample_rate, cam_delay)

    # TODO temporary
    data_dir = os.path.join("data", "2023-07-14-13-56-48")
    calibration.wav2dat(data_dir)

    # analysis part
    calibration.create_paramfile(data_dir, width, height, sample_rate, num_channels)

    analysis.dat2wav(data_dir, num_channels)
    # USV segmentation
    input(data_dir + "\n" + "Do USV segmentation and press Enter to continue...")
    #calibration.generateMicPosFile(mic_array_position[0])

    analysis.create_localization_video(data_dir, calib_path, color_eq=False)
