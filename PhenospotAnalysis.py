import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter
from pathlib import Path

import Config as root_path
import Functions as fn
from CreatePhasecontrastVideo import create_phase_contrast_video
from CreateResonantStack import create_res_map

t_interval = create_phase_contrast_video()
create_res_map()

'''################################################################'''
'''#   Data Analysis                                              #'''
'''################################################################'''

save_path = Path(root_path.root_path, root_path.exp_path)

''' Why do we remove a wave_step from wave_final'''
wav_ref = np.arange(root_path.wave_initial,
                    root_path.wave_final - root_path.wave_step,
                    root_path.wave_step)

''' Get ROI from user'''
rect_coords = fn.get_area(map_store[:, :, -1])

''' Slice resonant map to ROI'''
map_regions = map_store[rect_coords[0][1]:rect_coords[1][1],
                        rect_coords[0][0]:rect_coords[1][0],
                        :]

''' Take average resonant wavelength over ROI for each time step'''
av_spec = np.mean(map_regions, axis=(0, 1))


''' Display ROI of resonant map for last time step'''
fig = plt.figure(figsize=(7, 7))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Area to be measured')
plt.imshow(map_regions[:, :, -1],
           interpolation='bilinear',
           cmap='autumn')
plt.pause(1)
plt.close()

'''Disply resonant shift with time'''
fig = plt.figure(figsize=(7, 7))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Resonant wavelength of averaged area over time')
plt.plot(t_interval, av_spec, 'or')
plt.ylim((root_path.wave_initial, root_path.wave_final))
plt.xlabel('Time [h]')
plt.ylabel('Resonant wavelength [nm]')
plt.pause(1)
plt.close()

'''Display spectra from regions'''
bins = np.arange(root_path.wave_initial,
                 root_path.wave_final,
                 root_path.wave_step)

fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.suptitle('Spectra of regions with time')
sub_x = int(np.ceil(np.sqrt(len(t_interval))))
sub_y = int(np.ceil(len(t_interval) / sub_x))

for idx, t in enumerate(t_interval):
    sub = fig.add_subplot(sub_x, sub_y, idx + 1)
    sub.set_xlabel(f'Time interval: {t} h')
    sub.set_ylabel('Intensity (au)')
    sub.hist(np.ravel(map_regions[:, :, idx]),
             bins=bins,
             density=False,
             facecolor='g',
             alpha=0.75)
fig.tight_layout()
plt.pause(1)
plt.close()

'''Get further ROIs from user'''
num_of_ROIs = root_path.cell_num
roi_locs = fn.get_ROI_areas(map_store[:, :, -1], num_of_pts=num_of_ROIs)

print(f'ROI locations -> {str(Path(save_path, root_path.roirect_data))}')
np.save(Path(save_path, root_path.roirect_data), np.array(roi_locs))

'''Slice hyperspectral maps into ROI regions'''
roi_regions = list()
for r in roi_locs:
    temp = map_store[r[0][1]:r[1][1], r[0][0]:r[1][0], :]
    roi_regions.append(temp)

'''Calculate mean value of each region (i.e average resonant wavelength)
   at each time interval.  Shift all datapoints so that first datapoint starts
   at zero'''
av_shift = list()
for roi in roi_regions:
    av = np.mean(roi, axis=(0, 1))
    av_shift.append(av - av[0])

print(f'Resonant shifts -> {str(Path(save_path, root_path.av_shift_data))}')
np.save(Path(save_path, root_path.av_shift_data), np.array(av_shift))

'''Display kinematics from different ROIs'''
fig = plt.figure(figsize=(7, 7))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Kinematics of ROIs')
for idx, data in enumerate(av_shift):
    plt.plot(t_interval, data, label=f'ROI {idx}')

plt.xlabel('Time after activation [hours]')
plt.ylabel('Resonance wavlength [nm]')
plt.legend()
fig.tight_layout()
plt.savefig(Path(save_path, root_path.res_over_t_chart), format='png')
print(
    f'Resonant shifts chart -> {str(Path(save_path, root_path.res_over_t_chart))}')
plt.pause(1)
plt.close()

'''Produce video of hyperspectral images of ROIs'''
writer = FFMpegWriter(fps=2)
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.suptitle('Hyperspectral maps of ROIs')
plt.gca().axes.axis('off')
with writer.saving(fig, Path(save_path, root_path.HyS_video), dpi=100):
    for Hyp_idx, t in enumerate(t_interval):
        plt.suptitle(f'Hyperspectral maps of ROIs\nTime: {t:0.1f} h')
        for idx, r in enumerate(roi_locs):
            temp = map_store[r[0][1]:r[1][1], r[0][0]:r[1][0], Hyp_idx]

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
        print(f'Frame {Hyp_idx + 1} of {len(t_interval)} written')
print(f'Hyperspectral video -> {str(Path(save_path, root_path.HyS_video))}')
plt.close()

'''Plot histogram of cell count versus resonant wavlength.  Only take value
   from 5th time interval ???'''
# How many points do you want to consider for the average resonance shift
# over time?
fin = 5
hist_data = [a[fin] for a in av_shift]

print(f'Histogram data -> {str(Path(save_path, root_path.histogram_data))}')
np.save(Path(save_path, root_path.histogram_data), np.array(hist_data))

binwidth = 0.1
bins = np.arange(np.amin(hist_data), np.amax(hist_data) + binwidth, binwidth)

fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title('Number of cells per resonance shift')
plt.hist(hist_data, bins=bins, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Resonance shift [nm]')
plt.ylabel('# of cells')
plt.savefig(Path(save_path, root_path.histogram_chart), format='png')
print(f'Histogram chart -> {str(Path(save_path, root_path.histogram_chart))}')
plt.pause(1)
plt.close()


'''Plot boxplot of histogram data for improved data visualisation'''
legend_name = 'well 01 ROI av'  # Needs to be constructed not hard-written
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update(fn.FONT_PARAMS)
plt.title(legend_name)
plt.boxplot(hist_data)
plt.ylabel('Resonance shift [nm]')
plt.savefig(Path(save_path, root_path.boxplot_chart), format='png')
print(f'Boxplot chart -> {str(Path(save_path, root_path.boxplot_chart))}')
plt.show()
