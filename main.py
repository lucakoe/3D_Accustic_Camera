# TODO make documetation of recording setup
# camera calibration
import pickle
import shutil
import subprocess

import h5py
import cv2
import avsync
import os
import datetime
import recording
import usvcam_main.usvcam.analysis as analysis
import misc.camera_calib.camera_calib as cameracalib
import calibration
from PIL import Image, ImageTk
from tkinter import Tk, Label
import pyaudio

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

# General settings
usvseg_path = './usvcam_main/usvseg_plus/usvseg09r2_plus.exe'
data_path = './data'
recording_dir_i_name="mic"
audio_recording_out_filename = 'audio.wav'
audio_recording_dat_out_filename = 'snd.dat'
video_recording_out_filename = 'vid.mp4'
syncfile_filename = 'sync.csv'
output_filename = 'video_audio.mp4'
output_localization_video_filename = './vid.loc.mp4'
parameter_filename = 'param.h5'
camera_calibration_filename = 'cam_calibration.h5'
camera_calibration_path = os.path.join(data_path, camera_calibration_filename)
mic_calibration_filename='micpos.h5'
mic_calibration_path = os.path.join(data_path, mic_calibration_filename)
temp_path = os.path.join(data_path, 'temp')
mic_array_devices = [1, 2]  # device number of microphones in order of output files
camera_devices = [0, 1]  # number doesnt matter, just length; implemented for possible extension
mic_array_position = [[-0.065, 0.023, 0.0135], [-0.065, 0.023, 0.0135]]  # relative to camera
mic_array_layout_default = [[], [], [], [], [11, 4, 13, 2], [], [], [], [], [], [], [], [], [], [], [],
                            [8, 9, 6, 7, 10, 11, 4, 5, 12, 13, 2, 3, 14, 15, 0,
                             1]]  # layout of used channels depending on number of channels used
mic_array_layout = [mic_array_layout_default[num_channels],
                    mic_array_layout_default[
                        num_channels]]  # mic channels arranged from left top to right bottom (when looking from behind)
mic_array_calibration = False  # if set true, a new calibration for the calib_path gets initiated
camera_calibration = False  # if set true, a new calibration of the camera gets initiated


def record_audio_trigger(data_path):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Create a new directory for this recording
    recording_dir = os.path.join(data_path, timestamp)
    os.makedirs(recording_dir)
    recording.record(recording_dir, audio_recording_out_filename, syncfile_filename, trigger_device, fps,
                     mic_array_devices,
                     num_channels, sample_rate,
                     chunk, recording_dir_i_name=recording_dir_i_name)

    recording_dir_i_path=[]
    for i in range(len(mic_array_devices)):
        recording_dir_i_path.append(os.path.join(recording_dir,recording_dir_i_name+str(i)))
        print(os.path.join(recording_dir_i_path[i],audio_recording_out_filename))
        recording.rearrange_wav_channels(os.path.join(recording_dir_i_path[i],audio_recording_out_filename), mic_array_layout[i],
                                         os.path.join(recording_dir_i_path[i],audio_recording_out_filename))


    input("Stop Video Recording and press enter. File gets read from temp folder")

    video_recording_temp_path = []
    for i in range(len(camera_devices)):
        video_recording_temp_path.append(os.path.join(temp_path, modify_file_name(video_recording_out_filename, i)))

        while (not os.path.exists(video_recording_temp_path[i])):
            input(f"No video file {i} found in temp folder, please move it there and check if the name is correct")

        if os.path.exists(video_recording_temp_path[i]):
            shutil.move(video_recording_temp_path[i], os.path.join(recording_dir_i_path[i],video_recording_out_filename))
        else:
            print(f"File {i} not found. Cannot move the file.")

    output_path = []
    for i in range(min(len(camera_devices), len(mic_array_devices))):
        output_path.append(os.path.join(recording_dir, modify_file_name(output_filename, i)))

        avsync.combine_vid_and_audio(os.path.join(recording_dir_i_path[i],audio_recording_out_filename),
                                     os.path.join(recording_dir_i_path[i],video_recording_out_filename),
                                     os.path.join(recording_dir_i_path[i], syncfile_filename),
                                     os.path.join(recording_dir_i_path[i],output_filename), fps, sample_rate, cam_delay)

    return recording_dir


def modify_file_name(file_name, i):
    # Split the file name into name and extension parts
    name, extension = file_name.split('.')
    if i is 0:
        return file_name
    # Create the modified file name with the provided number
    modified_file_name = f"{name}_{i}.{extension}"

    return modified_file_name


def create_parameter_files(recording_dir):
    recording_dir_i_path = []
    for i in range(len(mic_array_devices)):
        recording_dir_i_path.append(os.path.join(recording_dir,recording_dir_i_name+str(i)))
    for i in range(min(len(camera_devices), len(mic_array_devices))):
        calibration.create_paramfile(recording_dir, width,
                                     height, sample_rate, num_channels,
                                     mic0pos=mic_array_position[i],
                                     camera_calibration_file=os.path.join(data_path,
                                                                          modify_file_name(camera_calibration_filename,
                                                                                           i)),
                                     paramfile_out_path=
                                     os.path.join(recording_dir_i_path[i], parameter_filename))


def conversion_and_segmentation(recording_dir):
    for i in range(len(mic_array_devices)):
        recording_dir_i_path = os.path.join(recording_dir, recording_dir_i_name + str(i))
        calibration.wav2dat(recording_dir, input_file_path=os.path.join(recording_dir_i_path, audio_recording_out_filename), output_file_path=os.path.join(recording_dir_i_path, audio_recording_dat_out_filename))
        analysis.dat2wav(recording_dir, num_channels, paramfile_path=os.path.join(recording_dir_i_path, parameter_filename),
                         audio_data_dat_path=os.path.join(recording_dir_i_path, audio_recording_dat_out_filename))
        # USV segmentation
        print(recording_dir + "\n" + "Do USV segmentation for microphone array " + str(
            i) + " and close USV Seg to continue...")
        subprocess.run([usvseg_path])

def calibrate_mic_array(recording_dir):
    for i in range(len(mic_array_devices)):
        recording_dir_i_path = os.path.join(recording_dir, recording_dir_i_name + str(i))
        SEG, P = calibration.my_pick_seg_for_calib(recording_dir_i_path)

        data = {"SEG": SEG, "P": P}

        # with open(modify_file_name('calibdata.pickle', i), 'wb') as f:
        #     pickle.dump(data, f)

        # with open(modify_file_name('calibdata.pickle',i), 'rb') as f:
        #     data = pickle.load(f)

        SEG = data["SEG"]
        P = data["P"]
        calibration.my_calc_micpos(recording_dir_i_path, SEG, P, h5f_outpath=os.path.join(recording_dir, modify_file_name(mic_calibration_filename, i)))
        user_input = input(
            "Do you want to save the calibration for microphone array " + str(i) + "? \nIf so type \"y\" otherwise type something else)")
        if user_input is "y":
            if os.path.exists(os.path.abspath(data_path)):
                shutil.move(os.path.join(recording_dir, modify_file_name(mic_calibration_filename, i)),
                            os.path.join(data_path, modify_file_name(mic_calibration_filename, i)))
            else:
                print("File or path not found")

    print("Mic Calibration finished")


def calibrate_cameras(recording_dir):
    recording_dir_i_path = []
    for i in range(len(mic_array_devices)):
        recording_dir_i_path.append(os.path.join(recording_dir,recording_dir_i_name+str(i)))

    for i in range(min(len(camera_devices),len(mic_array_devices))):
        video_path = os.path.join(recording_dir_i_path[i], video_recording_out_filename)
        print("\nStart camera "+str(i)+" calibration with: " + video_recording_out_filename)
        cameracalib.analyze_chessboardvid(video_path, 23, saveimg=True, frame_intv=5)
        print("Generating calibration file")
        cameracalib.calibrate_intrinsic(video_path)
        mtx = None
        dist = None
        with h5py.File(os.path.splitext(video_path)[0] + '_cam_intrinsic.h5', mode='r') as f:
            print(f.keys())
            mtx = f['/mtx'][()]
            dist = f['/dist'][()]

        print("CAMEAR MATRIX:")
        print(mtx)

        print("DISTORTION COEFFS:")
        print(dist)
        running = True

        try:
            if os.path.exists(os.path.abspath("./misc/camera_calib/tmp")):
                # Use os.startfile to open the Windows Explorer window at the specified path
                os.startfile(os.path.abspath("./misc/camera_calib/tmp"))
            else:
                print("Path does not exist.")
        except Exception as e:
            print("An error occurred:", e)

        while running:
            user_input = input(
                "Put in the number of picture in Camera Calibration Temp you want to undistort as a test. \nOtherwise just press Enter ")
            if user_input is "":
                running = False
            elif os.path.exists(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg")):
                img = cv2.imread(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg"))
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # undistort
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + '_calibresult.png'),
                            dst)

                # Open the images using PIL
                image_distorded = Image.open(
                    os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + ".jpg"))
                image_undistorded = Image.open(
                    os.path.join(os.path.abspath("./misc/camera_calib/tmp"), user_input + '_calibresult.png'))
                max_height = max(image_distorded.height,
                                 image_undistorded.height)  # Find the maximum height between the two images
                total_width = image_distorded.width + image_undistorded.width  # Calculate the total width required for both images side by side
                combined_image = Image.new('RGB', (
                    total_width, max_height))  # Create a new blank image to combine the two images
                combined_image.paste(image_distorded, (0, 0))  # Paste the first image on the left side
                combined_image.paste(image_undistorded,
                                     (image_distorded.width, 0))  # Paste the second image on the right side
                root = Tk()  # Show the combined image using tkinter
                root.title("Images Side by Side")

                # Convert the PIL image to a PhotoImage object to display in tkinter
                img_tk = ImageTk.PhotoImage(combined_image)

                # Create a label and set the image
                label = Label(root, image=img_tk)
                label.image = img_tk
                label.pack()

                root.mainloop()

            else:
                print("File does not exist")

        user_input = input(
            "Do you want to save the calibration for camera " + str(
                i) + "? \nIf so type \"y\" otherwise type something else)")
        if user_input is "y":
            if os.path.exists(os.path.abspath(data_path)):
                shutil.move(os.path.splitext(video_path)[0] + '_cam_intrinsic.h5',
                            os.path.join(data_path, modify_file_name(camera_calibration_filename, i)))
            else:
                print("File or path not found")
    print("Camera Calibration finished")

def create_localization_video(recording_dir):
    for i in range(min(len(camera_devices),len(mic_array_devices))):
        recording_dir_i_path = os.path.join(recording_dir, recording_dir_i_name + str(i))
        analysis.create_localization_video(recording_dir_i_path, os.path.join(data_path, modify_file_name(mic_calibration_filename,i)),
                                           paramfile_path=os.path.join(recording_dir_i_path, parameter_filename),
                                           syncfile_path=os.path.join(recording_dir_i_path, syncfile_filename),
                                           videofile_path=os.path.join(recording_dir_i_path, video_recording_out_filename),
                                           audio_data_dat_path=os.path.join(recording_dir_i_path, audio_recording_dat_out_filename),
                                           outfile_path=os.path.join(recording_dir_i_path, output_localization_video_filename), color_eq=False)


if __name__ == "__main__":
    # recording.get_microphone_info()
    # recording_dir = record_audio_trigger(data_path)
    recording_dir = r"C:\Users\experimenter\PycharmProjects\Accustic_Camera\data\2023-07-28-17-16-48"

    if camera_calibration:
        calibrate_cameras(recording_dir)
    create_parameter_files(recording_dir)
    conversion_and_segmentation(recording_dir)
    if mic_array_calibration:
        calibrate_mic_array(recording_dir)
    create_localization_video(recording_dir)

