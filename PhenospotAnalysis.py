from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
from PIL import Image
from datetime import datetime

import Settings
import Functions as fn

root_path = Path(Settings.root_path)
exp_path = Path(Settings.exp_path)

HyS_path = Path(Settings.HyS_path)
PhC_path = Path(Settings.PhC_path)
map_savepath = Path(f'{Settings.savepath}/{Settings.mapfilename}')
savepath = Path(Settings.savepath)

PhC_data_paths = [p for p in PhC_path.glob('*.tiff')]
PhC_data_paths.sort(key=lambda x: x.parts[-1])

HyS_data_paths = [p for p in HyS_path.iterdir() if p.is_dir()]
HyS_data_paths.sort(key=lambda x: int(x.parts[-1]))

temp_im = Image.open(PhC_data_paths[0])

nfiles = len(PhC_data_paths)
all_data = list()
basenames = list()
res_wav = list()
time_elapsed = np.arange(len(HyS_data_paths))
map_store = np.zeros((temp_im.size[1], temp_im.size[0], len(HyS_data_paths)))

# print(f'{time_elapsed=}, {res_wav=}, {map_store=}')
# im.show()

mod_time = list()
for PhC_fp in PhC_data_paths:
    m_timestamp = PhC_fp.stat().st_mtime
    mod_time.append(datetime.fromtimestamp(m_timestamp))

t_interval = [((t - mod_time[0]).seconds/3600) +
              Settings.act_time for t in mod_time]

# fn.save_phasecontrast_video(t_interval, PhC_data_paths)

# Why do we remove a wave_step from wave_final
wav_ref = np.arange(Settings.wave_initial,
                    Settings.wave_final - Settings.wave_step,
                    Settings.wave_step)
xvals = np.linspace(0, len(wav_ref), 1000)
wav_int = np.interp(xvals, range(len(wav_ref)), wav_ref)
peak = True

# H_idx = 0
# HyS_fp = '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/Hyperspectral/1'
for H_idx, HyS_fp in enumerate(HyS_data_paths):

    # imstack = fn.build_image_stack(HyS_fp)

    # if peak:
    #     resonance_indexes = np.argmax(imstack, axis=2)
    # else:
    #     resonance_indexes = np.argmin(imstack, axis=2)

    # map_store[:, :, H_idx] = wav_ref[resonance_indexes]
    
    ############## TEMP CODE FOR TESTING TO SAVE LOADING IMAGES #############
    temp_filepath = Path(
        '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/test_map_store.npy')
    map_store = np.load(temp_filepath, mmap_mode='r')
    temp_filepath = Path(
        '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/test_imstack.npy')
    imstack = np.load(temp_filepath, mmap_mode='r')
    #########################################################################
    
    print(f'Resonant map {H_idx} of {len(HyS_data_paths)} complete')

    roi_locs = fn.get_Pts(map_store[:, :, 0], num_of_pts=1,
                          ROI_size=(100, 100), ROI=False)

    y_min = min(roi_locs[0][1], roi_locs[1][1])
    y_max = max(roi_locs[0][1], roi_locs[1][1])
    x_min = min(roi_locs[0][0], roi_locs[1][0])
    x_max = max(roi_locs[0][0], roi_locs[1][0])

    region = imstack[y_min:y_max, x_min:x_max, :]  # slice imstack to ROI 
    mapregion = map_store[y_min:y_max, x_min:x_max, 0]  # slice mapstore
    av_spec = np.mean(region, axis=(0, 1))  # take mean of each 2D region
    res_wav.append(wav_ref[np.argmax(av_spec)])  # find peak of spectrum

    ''' Display area that will be measured'''
    fig = plt.figure(figsize=(7, 7))
    font_params = {'figure.titlesize': 20,
                'axes.titlesize': 15,
                'axes.labelsize': 15,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12}
    plt.rcParams.update(font_params)
    plt.title('Area to be measured', fontsize=16)
    plt.imshow(mapregion,
               interpolation='bilinear',
               cmap='autumn')
    plt.pause(1)
    plt.close(fig)

    '''Display spectrum from analysed area'''
    fig = plt.figure(figsize=(7, 7))
    font_params = {'figure.titlesize': 20,
                   'axes.titlesize': 15,
                   'axes.labelsize': 15,
                   'xtick.labelsize': 12,
                   'ytick.labelsize': 12,
                   'legend.fontsize': 12}
    plt.rcParams.update(font_params)
    plt.title('Spectrum of averaged area', fontsize=16)
    plt.plot(wav_ref, av_spec, 'or')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance [a.u.]')
    plt.show()

    break

print(f'{res_wav=}')
np.save(map_savepath, map_store)
