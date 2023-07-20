import os
import numpy as np
import pyaudio
import wave
import csv
import time
import threading
import nidaqmx
from nidaqmx.constants import LineGrouping

# Global variable for current audio frame
current_audio_frame_numbers = []
current_video_frame = 0
audio_frames = []
start_time = 0
fps = 0


def record_audio(stream, mic_i, chunk_size, stop_event):
    global audio_frames, current_audio_frame_numbers, sample_rate, start_time
    while not stop_event.is_set():
        # Capture audio frames from each stream
        audio_frames[mic_i].append(stream.read(chunk_size))
        current_audio_frame_numbers[mic_i] += 1


def record_video_trigger(trigger_device, fps, csv_writer, stop_event, chunk_size):
    global current_audio_frame_numbers, current_video_frame, sample_rate, start_time
    start_time = time.time()
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(
            trigger_device, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
        )

        while not stop_event.is_set():
            # Calculate the expected timestamp of the current video frame
            expected_audio_time = current_video_frame / fps

            # Wait until the expected timestamp is reached
            while (time.time() - start_time) < expected_audio_time and not stop_event.is_set():
                time.sleep(0.001)

            if stop_event.is_set():
                break

            new_row = [time.time() - start_time]
            for current_audio_frame_number in current_audio_frame_numbers:
                new_row.append(current_audio_frame_number * chunk_size)

            task.write([True])
            task.write([False])
            print("time: ", time.time() - start_time)

            # Save the timestamps to the CSV file
            csv_writer.writerow(new_row)
            current_video_frame += 1



def record(data_dir, audio_recording_out_filename, syncfile_filename, trigger_device, fps, mic_array_devices,
           num_channels, sample_rate,
           chunk_size):

    get_trigger_info()
    get_microphone_info(mic_array_devices)

    input("Press Enter to start recording. Don't forget to start the video recording")

    print("Start recording\n")
    # Set the paths for the output files
    audio_recording_out_path = os.path.join(data_dir, audio_recording_out_filename)
    syncfile_path = os.path.join(data_dir, syncfile_filename)

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open audio streams for recording from each microphone
    streams = []
    for mic_array_device in mic_array_devices:
        stream = audio.open(format=pyaudio.paInt16,
                            channels=num_channels,
                            rate=sample_rate,
                            input=True,
                            input_device_index=mic_array_device,  # use the correct microphone device index here
                            frames_per_buffer=chunk_size)
        streams.append(stream)
        current_audio_frame_numbers.append(0)
        audio_frames.append([])

    # Initialize CSV writer
    with open(syncfile_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(['Time (s)', 'Video Frame Number', 'Audio Frame Number', 'Audio Position in file'])

        # Start recording audio in a separate thread

        stop_event = threading.Event()
        audio_threads = []
        for i in range(len(mic_array_devices)):
            audio_threads.append(threading.Thread(target=record_audio, args=(
                streams[i], i, chunk_size, stop_event)))
            audio_threads[i].start()

        # Start recording video in a separate thread
        record_event = threading.Event()
        video_thread = threading.Thread(target=record_video_trigger,
                                        args=(trigger_device, fps, csv_writer, stop_event, chunk_size))
        video_thread.start()

        # Wait for the user to stop recording
        input("Press Enter to stop recording\n")

        # Stop recording and join threads
        stop_event.set()
        record_event.set()
        for audio_thread in audio_threads:
            audio_thread.join()
        video_thread.join()

        # Close audio streams and PyAudio
        for stream in streams:
            stream.stop_stream()
            stream.close()
        audio.terminate()

        # Save the recorded audio to WAV files for each microphone
        for i in range(len(mic_array_devices)):
            if i == 0:
                audio_recording_out_path_i = audio_recording_out_path
            else:
                audio_recording_out_path_i = f"{audio_recording_out_path[:-4]}_{i}.wav"

            wave_file = wave.open(audio_recording_out_path_i, "wb")
            wave_file.setnchannels(num_channels)
            wave_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(b"".join(audio_frames[i]))
            wave_file.close()

        # Check the length of the audio and video data arrays
        for audio_frame in audio_frames:
            print("Length of audio data:", len(audio_frame))
        print("Length of video data:", current_video_frame)

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

def get_microphone_info(array_of_mics=None):
    p = pyaudio.PyAudio()
    if array_of_mics is None:
        num_devices = p.get_device_count()
        print(f"Number of Audio Devices (Microphones): {num_devices}\n")
        array_of_mics=range(num_devices)

    for mic_device in array_of_mics:

        device_info = p.get_device_info_by_index(mic_device)
        print(f"Device Index: {mic_device}")
        print(f"    Name: {device_info['name']}")
        print(f"    Max Input Channels: {device_info['maxInputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']} Hz")

    p.terminate()


def get_trigger_info():
    try:
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(
                "Dev1/port1/line0", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )

            task.write([False])

    except nidaqmx.DaqError as e:
        print(e)
    print("Trigger Device: " + trigger_device)
