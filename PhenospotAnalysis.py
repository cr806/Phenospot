import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter
from pathlib import Path
from PIL import Image
from datetime import datetime

import Settings
import Functions as fn


HyS_path = Path(Settings.HyS_path)
PhC_path = Path(Settings.PhC_path)
map_savepath = Path(f'{Settings.mapfilename}')
roi_savepath = Path(f'{Settings.roirectfilename}')
av_shift_savepath = Path(f'{Settings.av_shiftfilename}')

PhC_data_paths = [p for p in PhC_path.glob('*.tiff')]
PhC_data_paths.sort(key=lambda x: x.parts[-1])

HyS_data_paths = [p for p in HyS_path.iterdir() if p.is_dir()]
HyS_data_paths.sort(key=lambda x: int(x.parts[-1]))

temp_im = Image.open(PhC_data_paths[0])
# fig = plt.figure(figsize=(7, 7))
# plt.rcParams.update(fn.FONT_PARAMS)
# plt.imshow(temp_im)
# plt.show()

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

fn.save_phasecontrast_video(t_interval, PhC_data_paths)

# Why do we remove a wave_step from wave_final
wav_ref = np.arange(Settings.wave_initial,
                    Settings.wave_final - Settings.wave_step,
                    Settings.wave_step)
xvals = np.linspace(0, len(wav_ref), 1000)
wav_int = np.interp(xvals, range(len(wav_ref)), wav_ref)

# H_idx = 0
# HyS_fp = '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab
# /data/Location_1/Hyperspectral/1'
for H_idx, HyS_fp in enumerate(HyS_data_paths):

    imstack = fn.build_image_stack(HyS_fp)
    resonance_indexes = fn.get_resonance_idxs(imstack)
    map_store[:, :, H_idx] = wav_ref[resonance_indexes]
    print(f'Resonant map {H_idx} of {len(HyS_data_paths)} complete')

    rect_coords = fn.get_area(map_store[:, :, 0])

    y_min = min(rect_coords[0][1], rect_coords[1][1])
    y_max = max(rect_coords[0][1], rect_coords[1][1])
    x_min = min(rect_coords[0][0], rect_coords[1][0])
    x_max = max(rect_coords[0][0], rect_coords[1][0])

    region = imstack[y_min:y_max, x_min:x_max, :]       # slice imstack to ROI
    mapregion = map_store[y_min:y_max, x_min:x_max, 0]  # slice mapstore
    av_spec = np.mean(region, axis=(0, 1))              # take mean of regions
    res_wav.append(wav_ref[np.argmax(av_spec)])         # find peak of spectrum

    ''' Display area that will be measured'''
    fig = plt.figure(figsize=(7, 7))
    plt.rcParams.update(fn.FONT_PARAMS)
    plt.title('Area to be measured')
    plt.imshow(mapregion,
               interpolation='bilinear',
               cmap='autumn')
    plt.pause(1)
    plt.close(fig)

    '''Display spectrum from analysed area'''
    fig = plt.figure(figsize=(7, 7))
    plt.rcParams.update(fn.FONT_PARAMS)
    plt.title('Spectrum of averaged area')
    plt.plot(wav_ref, av_spec, 'or')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance [a.u.]')
    plt.pause(1)
    plt.close(fig)


print(f'{map_store} -> {map_savepath}')
np.save(map_savepath, map_store)

# ############# TEMP CODE FOR TESTING TO SAVE LOADING IMAGES #############
# temp_filepath = Path(
#     '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/test_map_store.npy')
# map_store = np.load(temp_filepath, mmap_mode='r')
# temp_filepath = Path(
#     '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/test_imstack.npy')
# imstack = np.load(temp_filepath, mmap_mode='r')
# ########################################################################
# temp_filepath = Path(
#     '/Volumes/krauss/Isabel/Phenospot_TF_data_science_support/Matlab/data/Location_1/test_roirect.npy')
# roi_locs = np.load(temp_filepath, mmap_mode='r')
# roi_locs = roi_locs.tolist()
# #########################################################################

'''Get ROI locations from user'''
num_of_ROIs = Settings.cell_num
roi_locs = fn.get_ROIs(map_store[:, :, 0], num_of_pts=num_of_ROIs,
                       ROI_size=Settings.ROI_size)

print(f'{np.array(roi_locs)} -> {roi_savepath}')
np.save(roi_savepath, np.array(roi_locs))

'''Slice hyperspectral maps into ROI regions'''
roi_regions = list()
for r in roi_locs:
    temp = map_store[r[1]:r[1] + Settings.ROI_size[1],
                     r[0]:r[0] + Settings.ROI_size[0], :]
    roi_regions.append(temp)

'''Calculate mean value of each region (i.e average resonant wavelength)
   at each time interval.  Shift all datapoints so that first datapoint starts
   at zero'''
av_shift = list()
for roi in roi_regions:
    av = np.mean(roi, axis=(0, 1))
    av_shift.append(av - av[0])

print(f'{np.array(av_shift)} -> {av_shift_savepath}')
np.save(av_shift_savepath, np.array(av_shift))

'''Display kinematics from different ROIs'''
fig = plt.figure(figsize=(7, 7))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Kinematics of ROIs')
for idx, data in enumerate(av_shift):
    plt.plot(t_interval, data, label=f'ROI {idx}')

plt.xlabel('Time after activation [hours]')
plt.ylabel('Resonance wavlength [nm]')
plt.legend()
plt.savefig(Settings.res_over_t_name, format='png')
plt.pause(1)
plt.close()

'''Produce video of hyperspectral images of ROIs'''
writer = FFMpegWriter(fps=2)
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.suptitle('Hyperspectral maps of ROIs')
plt.gca().axes.axis('off')
with writer.saving(fig, Settings.HyS_videoname, dpi=100):
    for Hyp_idx, t in enumerate(t_interval):
        print(f'Hyperspectral image {Hyp_idx} processed')
        plt.suptitle(f'Hyperspectral maps of ROIs\nTime: {t:0.1f} h')
        for idx, r in enumerate(roi_locs):
            temp = map_store[r[1]:r[1] + Settings.ROI_size[1],
                             r[0]:r[0] + Settings.ROI_size[0], Hyp_idx]

            sub_x = int(np.ceil(np.sqrt(num_of_ROIs)))
            sub_y = int(np.ceil(num_of_ROIs / sub_x))
            sub = fig.add_subplot(sub_x, sub_y, idx + 1)
            sub.axes.get_xaxis().set_ticks([])
            sub.axes.get_yaxis().set_ticks([])
            sub.set_xlabel(f'ROI: {idx}')
            sub.imshow(temp,
                       interpolation='bilinear',
                       cmap='autumn')
        writer.grab_frame()

'''Plot histogram of cell count versus resonant wavlength.  Only take value
   from 5 time interval ???'''
# How many points do you want to consider for the average resonance shift
# over time?
fin = 5
hist_data = [a[fin] for a in av_shift]
binwidth = 0.1
bins = np.arange(np.amin(hist_data), np.amax(hist_data) + binwidth, binwidth)

fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Number of cells per resonance shift')
plt.hist(hist_data, bins=bins, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Resonance shift [nm]')
plt.ylabel('# of cells')
plt.savefig(Settings.histogram_name, format='png')
plt.pause(1)
plt.close()

print(f'{np.array(av_shift)} -> {av_shift_savepath}')
np.save(Settings.histogram_workspace, np.array(hist_data))

'''Plot boxplot of histogram data for improved data visualisation'''
legend_name = 'well 01 ROI av'  # Needs to be constructed not hard-written
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title(legend_name)
plt.boxplot(hist_data)
plt.ylabel('Resonance shift [nm]')
plt.savefig(Settings.boxplot_name, format='png')
plt.show()
