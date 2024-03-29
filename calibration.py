import os
from pydub import AudioSegment
from pydub.generators import Sine
import h5py
import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.io
import scipy.io.wavfile
import cv2
import math
import usvcam_main.usvcam.tool as tool
import copy

import glob
import pickle
import sklearn
from tqdm import tqdm


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
    [0.042 * 1, 0.0, 0.0],  # Mic 12
    [0.0, 0.042 * 1, 0.0],  # Mic 3
    [0.042 * 1, 0.042 * 1, 0.0]]  # Mic 5
                       ):
    # generates arrangement matrix for 16 channels
    mic_array_arrangement = []
    for x in range(4):
        for y in range(4):
            mic_array_arrangement.append([x * 0.042, y * 0.042])

    # Create datasets
    result = np.add(np.array(mic_array_arrangement), np.array(mic_array_position))

    result = result / 1000

    with h5py.File("./data/micpos.h5", mode='w') as f:
        f.create_dataset('/result/micpos', data=result)


def checkMicPosFile(path):
    with h5py.File(path, 'r') as f:
        print(f.keys())
        print(f['result/micpos'][()])


def wav2dat(data_dir, input_file_path=None, output_file_path=None):
    if input_file_path is None:
        input_file_path = data_dir + '/audio.wav'
    if output_file_path is None:
        output_file_path = data_dir + '/snd.dat'

    A = scipy.io.wavfile.read(input_file_path)
    max_int16 = 32767
    max_int32 = 2147483647
    X = A[1]

    with open(output_file_path, 'wb') as f:
        f.write(X.tobytes())


def create_paramfile(data_dir, image_width=640, image_height=768,
                     daq_fs=192000,
                     daq_n_ch=4,
                     camera_height=2.0, mic0pos=[0, 0, 0], camera_calibration_file=None, paramfile_out_path=None):
    pressure_calib_array = []
    speedOfSound = 343.0
    camera_matrix = None
    if paramfile_out_path is None:
        paramfile_out_path = data_dir + '/param.h5'

    coeff = np.array([0, 0, 0, 0, 0], dtype=float)
    if camera_calibration_file is not None:
        with h5py.File(camera_calibration_file, mode='r') as f:
            print(f.keys())
            camera_matrix = f['/mtx'][()]
            coeff = f['/dist'][()]
    else:
        camera_fov_y = 1.047198
        fy = image_height / (2 * np.tan(camera_fov_y / 2))
        fx = fy
        ppx = image_width / 2
        ppy = image_height / 2
        camera_matrix = [[fx, 0., ppx][0., fy, ppy][0., 0., 1.]]

    fx, fy, ppx, ppy = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]
    for channels in range(daq_n_ch):
        pressure_calib_array.append(1)
    pressure_calib = np.array(pressure_calib_array, dtype=float)
    mic0pos = np.array(mic0pos, dtype=float)
    r = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    t = np.array([0, 0, 0], dtype=float)

    with h5py.File(paramfile_out_path, mode='w') as f:
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


def my_pick_seg_for_calib(data_dir, paramfile_path=None, vidfile_path=None, syncfile_path=None, n_pos_check=20):
    print('(some instruction here)')

    if paramfile_path is None:
        paramfile_path = data_dir + '/param.h5'
    if vidfile_path is None:
        vidfile_path = data_dir + '/vid.mp4'
    if syncfile_path is None:
        syncfile_path = data_dir + '/sync.csv'

    vr = cv2.VideoCapture(vidfile_path)

    with h5py.File(paramfile_path, mode='r') as f:
        fs = f['/daq_param/fs'][()]
        n_ch = f['/daq_param/n_ch'][()]

    T = np.genfromtxt(syncfile_path, delimiter=',')

    seg = tool.load_usvsegdata_ss(data_dir)
    _, I_ss = np.unique(seg[:, 4], axis=0, return_inverse=True)
    n_ss = np.max(I_ss) + 1

    # snout pos of each call
    P = []
    SEG = []
    L = glob.glob(data_dir + '/*.usvseg_dat.csv')
    wav_name = os.path.splitext(os.path.splitext(os.path.basename(L[0]))[0])[0]

    def update_disp():
        disp_img2 = copy.deepcopy(disp_img)
        if not np.isnan(p_crnt[0]):
            cv2.circle(disp_img2, (int(p_crnt[0]), int(p_crnt[1])), color=(0, 0, 255), thickness=2, radius=5)
        for p in P:
            cv2.drawMarker(disp_img2, (int(p[0]), int(p[1])), color=(0, 255, 255), thickness=2, markerSize=5)

        cv2.imshow(wname, disp_img2)

    def onMouse(event, x, y, flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            p_crnt[0] = x
            p_crnt[1] = y
            update_disp()

    wname = 'click sound source'
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, onMouse, [])
    disp_img = np.zeros([int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vr.get(cv2.CAP_PROP_FRAME_WIDTH)), 3], np.uint8)
    p_crnt = np.zeros([2], float)
    p_crnt[:] = np.nan
    for i_ss in range(n_ss):
        seg2 = seg[I_ss == i_ss, :]
        i_frame = tool.time2vidframe(data_dir, (np.min(seg2[:, 0]) + np.max(seg2[:, 0])) / 2, T, fs)
        vr.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, frame = vr.read()

        imgfile = data_dir + '/seg/' + wav_name + '_{:04}.jpg'.format(int(seg2[0, 4]))
        img = cv2.imread(imgfile)

        disp_img = frame
        r = 100
        a = cv2.resize(img, (r, r))
        disp_img[:r, -r:] = a
        update_disp()

        k = cv2.waitKey()
        if k == 27:  # ESC
            break

        if not np.isnan(p_crnt[0]):
            P.append(copy.deepcopy(p_crnt))
            SEG.append(seg2)

        p_crnt[:] = np.nan

    cv2.destroyAllWindows()

    return SEG, P


def my_calc_micpos(data_dir, SEG, P, calibfile=None, audio_data_dat_path=None, n_ch=16, h5f_outpath=None):
    if audio_data_dat_path is None:
        audio_data_dat_path = os.path.join(data_dir, './snd.dat')

    paramfile = data_dir + '/param.h5'
    with h5py.File(paramfile, mode='r') as f:
        fs = f['/daq_param/fs'][()]
        n_ch = f['/daq_param/n_ch'][()]
        speedOfSound = f['/misc/speedOfSound'][()]
        pressure_calib = f['/misc/pressure_calib'][()]
        mic0pos = f['/misc/mic0pos'][()]


    with open(audio_data_dat_path, 'rb') as fp_dat:

        def get_error(dx, P, SEG, data_dir, mic0pos, speedOfSound):

            dx = np.reshape(dx, (n_ch - 1, 3))
            dx = np.vstack([np.array([0, 0, 0]), dx])
            dx = dx + mic0pos

            n_pos_check = len(SEG)
            pwr = np.zeros(n_pos_check)

            for i_seg in range(n_pos_check):
                seg2 = copy.deepcopy(SEG[i_seg])

                pt = np.array([P[i_seg]])

                tau = tool.get_tau(data_dir, None, speedOfSound, points=pt, micpos=dx)

                S = tool.calc_seg_power(fp_dat, seg2, tau, fs, n_ch, pressure_calib, return_average=True)

                pwr[i_seg] = S

            e = -np.sum(pwr ** 2)

            return e

        global Nfeval
        Nfeval = 1

        def callbackF(Xi):
            global Nfeval
            e = get_error(Xi, P, SEG, data_dir, mic0pos, speedOfSound)
            print('iter:{0:4d}, f(x) = '.format(Nfeval) + str(-e))
            Nfeval += 1
            # p = np.reshape(Xi, (n_ch - 1, 3))
            # p = np.vstack([np.array([0, 0, 0]), p])
            # print(p + mic0pos)

        # run optimization
        if calibfile is None:
            # dx0 = np.tile([0,0,0], (n_ch-1,1))
            # initial mic position relative to mic ch14 (mic0pos)

            if float(math.sqrt(n_ch)).is_integer():
                # generates arrangement matrix for 16 channels
                dx0 = []
                for x in range(int(math.sqrt(n_ch))):
                    for y in range(int(math.sqrt(n_ch))):
                        dx0.append([x * 0.042, y * 0.042, 0])
                dx0 = np.array(dx0)
                dx0 = dx0[1:, :] - dx0[0, :]
                print(dx0)
            else:
                print("Provide calibfile")
        else:
            with h5py.File(calibfile, mode='r') as f:
                dx0 = f['/result/micpos'][()]
            dx0 = dx0[1:, :] - dx0[0, :]

        # lower and upper bound of search
        # lb = dx0 - 0.005
        # ub = dx0 + 0.005
        lb = np.tile([-0.1, 0.0, -0.01], (n_ch - 1, 1))
        ub = np.tile([0.1, 0.2, 0.01], (n_ch - 1, 1))

        dx0 = dx0.flatten()
        lb = lb.flatten()
        ub = ub.flatten()

        b = scipy.optimize.Bounds(lb, ub)

        print('Running optimization...')
        R = scipy.optimize.minimize(get_error, x0=dx0, args=(P, SEG, data_dir, mic0pos, speedOfSound),
                                    method='L-BFGS-B', bounds=b, callback=callbackF, options={'maxiter': 50})

        dx_pred = R.x
        dx_pred = np.reshape(dx_pred, (n_ch - 1, 3))
        dx_pred = np.vstack([np.array([0, 0, 0]), dx_pred])
        micpos = dx_pred + mic0pos
        print('estimated micpos:')
        print(micpos)

        if h5f_outpath is None:
            h5f_outpath = data_dir + '/micpos.h5'
        with h5py.File(h5f_outpath, mode='w') as f:
            f.create_dataset('/result/micpos', data=micpos)
            for i_seg in range(len(SEG)):
                f.create_dataset('/seg/seg{:06}'.format(i_seg), data=SEG[i_seg])
                f.create_dataset('/snout_pos/seg{:06}'.format(i_seg), data=P[i_seg])

        print('the result is saved in: ' + h5f_outpath)


if __name__ == '__main__':
    # generateMicPosFile([13, -35, 8])
    # checkMicPosFile('usvcam_main/test_data/micpos.h5')
    # checkMicPosFile('usvcam_main/test_data/micpos_custom.h5')
    checkMicPosFile((r'./data\2023-07-28-15-23-45\micpos.h5'))
    # wav2dat("./data/2023-07-11-15-14-36")
    # create_paramfile("./data/2023-07-12-10-27-46",640,480,48000,4)
