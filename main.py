# TODO make documetation of recording setup
# camera calibration
# 2 mic arrays
# camera focus
# (restructure program)

import shutil
import avsync
import os
import datetime
import recording
import usvcam_main.usvcam.analysis as analysis
import calibration
import pyaudio

# General settings
audio_recording_out_filename = 'audio.wav'
video_recording_out_filename = 'vid.mp4'
syncfile_filename = 'sync.csv'
output_filename = 'video_audio.mp4'
parameter_filename = 'param.h5'
calib_path = './data/micpos.h5'
temp_path = './data/temp'
mic_array_devices = [1, 2]  # number of microphones
mic_array_position = [[-0.063, 0.032, 0.005], [0, 0, 0]]  # relative to camera
mic_array_layout = [[8, 9, 6, 7, 10, 11, 4, 5, 12, 13, 2, 3, 14, 15, 0,
                     1], [8, 9, 6, 7, 10, 11, 4, 5, 12, 13, 2, 3, 14, 15, 0,
                          1]]  # mic channels arranged from left top to right bottom
new_calibration = False  # if set true, a new calibration for the calib_path gets initiated

# Audio settings
num_channels = 16
sample_rate = 44100  # careful use the same as the audio device in the system settings
chunk = 1024

# Video settings
trigger_device = "Dev1/port1/line0"
width = 640
height = 480
fps = 20
cam_delay = 0.0

def record_and_analyze():
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Create a new directory for this recording
    data_dir = os.path.join("data", timestamp)
    os.makedirs(data_dir)
    recording.record(data_dir, audio_recording_out_filename, syncfile_filename, trigger_device, fps, mic_array_devices,
                     num_channels, sample_rate,
                     chunk)

    audio_recording_out_path = []
    for i in range(len(mic_array_devices)):
        if i == 0:
            audio_recording_out_path.append(os.path.join(data_dir, audio_recording_out_filename))
        else:
            audio_recording_out_path.append( f"{os.path.join(data_dir, audio_recording_out_filename)[:-4]}_{i}.wav")

        recording.rearrange_wav_channels(audio_recording_out_path[i], mic_array_layout[i],
                                         audio_recording_out_path[i])

    # USV segmentation
    input("Stop Video Recording and press enter. File gets read from temp folder")
    while (not os.path.exists(os.path.join(temp_path, video_recording_out_filename))):
        input("No video file found in temp folder, please move it there and check if the name is correct")
    if os.path.exists(os.path.join(temp_path, video_recording_out_filename)):
        shutil.move(os.path.join(temp_path, video_recording_out_filename),
                    os.path.join(data_dir, video_recording_out_filename))
    else:
        print("File not found. Cannot move the file.")

    avsync.combine_vid_and_audio(os.path.join(data_dir, audio_recording_out_filename),
                                 os.path.join(data_dir, video_recording_out_filename),
                                 os.path.join(data_dir, syncfile_filename),
                                 os.path.join(data_dir, output_filename), fps, sample_rate, cam_delay)

    # # TODO temporary
    # data_dir = os.path.join("data", "2023-07-14-13-56-48")
    # calibration.wav2dat(data_dir)
    #
    # # analysis part
    # calibration.create_paramfile(data_dir, width, height, sample_rate, num_channels)
    # analysis.dat2wav(data_dir, num_channels)
    # # USV segmentation
    # input(data_dir + "\n" + "Do USV segmentation and press Enter to continue...")
    #
    # if new_calibration:
    #     SEG, P = calibration.my_pick_seg_for_calib(data_dir)
    #
    #     data = {"SEG": SEG, "P": P}
    #
    #     # with open('calibdata.pickle','wb') as f:
    #     #      pickle.dump(data, f)
    #     #
    #     # with open('calibdata.pickle', 'rb') as f:
    #     #     data = pickle.load(f)
    #
    #     SEG = data["SEG"]
    #     P = data["P"]
    #     calibration.my_calc_micpos(data_dir, SEG, P, h5f_outpath='./micpos.h5')
    #
    # analysis.create_localization_video(data_dir, calib_path, color_eq=False)





if __name__ == "__main__":
    #recording.get_microphone_info()
    record_and_analyze()
