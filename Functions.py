import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredText

import Settings

FONT_PARAMS = {'figure.titlesize': 20,
               'axes.titlesize': 15,
               'axes.labelsize': 15,
               'xtick.labelsize': 12,
               'ytick.labelsize': 12,
               'legend.fontsize': 12}


def save_phasecontrast_video(t_interval, PhC_data_paths):
    writer = FFMpegWriter(fps=2)
    fig, ax = plt.subplots(1, 1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    with writer.saving(fig, Settings.videoname, dpi=100):
        for t, PhC_fp in zip(t_interval, PhC_data_paths):
            temp = np.array(Image.open(PhC_fp))
            temp = np.flipud(temp)
            temp = temp / np.amax(temp)
            temp = sp.signal.medfilt2d(temp, kernel_size=3)
            # all_data.append(temp)

            try:
                img.set_data(temp)
            except NameError:
                img = ax.imshow(temp, cmap='gray')
                img.set_clim(vmin=0, vmax=1)

            at = AnchoredText(f'Time: {t:0.1f} h',
                              prop=dict(size=20),
                              frameon=True,
                              loc='lower left')
            at.patch.set_boxstyle('square, pad=0.1')
            ax.add_artist(at)

            writer.grab_frame()


def build_image_stack(HyS_fp):
    data_paths = [h for h in Path(HyS_fp).glob('*.tiff')]

    # This is BAD, filename should be standardised or use delimiters
    data_paths.sort(key=lambda x: int(x.stem[6:]))
    temp_im = Image.open(data_paths[0])
    imstack = np.zeros((temp_im.size[1], temp_im.size[0], len(data_paths)))

    for d_idx, d in enumerate(data_paths):
        with Image.open(d) as im:
            imstack[:, :, d_idx] = im
            print(f'Imported image {d_idx} of {len(data_paths)}')

    # Results in a 2.2Gb file, larger than images when stored separately
    # np.save(f'imstack_{H_idx}.npy', imstack)

    return imstack


def get_resonance_idxs(imstack, peak=True):
    if peak:
        return np.argmax(imstack, axis=2)
    else:
        return np.argmin(imstack, axis=2)

def get_Pts(data_image, num_of_pts=4, ROI_size=(100, 100), ROI=True):
    if not ROI:
        num_of_pts = 2
    fig, ax = plt.subplots(1, 1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    img = ax.imshow(data_image,
                    interpolation='bilinear',
                    cmap='autumn')
    # img.set_clim(vmin=Settings.wave_initial,
    #              vmax=Settings.wave_final - Settings.wave_step)
    bar = plt.colorbar(img)
    bar.set_label('Resonant Wavelength [nm]')

    locs = []
    while True:
        plt.title(f'You have selected {len(locs)} / {num_of_pts}',
                  fontsize=16)
        if ROI:
            for pt in locs:
                rect = patches.Rectangle(pt,
                                         ROI_size[0],
                                         ROI_size[1],
                                         linewidth=2,
                                         edgecolor='w',
                                         facecolor='none')
                ax.add_patch(rect)
        plt.draw()
        if len(locs) >= num_of_pts:
            break
        pt = plt.ginput(1, timeout=-1)[0]
        if ROI:
            pt = (pt[0] - ROI_size[0]/2, pt[1] - ROI_size[1]/2)
        locs.append((int(pt[0]), int(pt[1])))

    if ROI:
        plt.pause(0.5)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 10))  # width, height in inches
        plt.title('ROIs to be measured', fontsize=16)
        plt.gca().axes.axis('off')
        for i in range(num_of_pts):
            sub_x = int(np.ceil(np.sqrt(num_of_pts)))
            sub_y = int(np.ceil(num_of_pts / sub_x))
            sub = fig.add_subplot(sub_x, sub_y, i + 1)
            sub.axes.get_xaxis().set_ticks([])
            sub.axes.get_yaxis().set_ticks([])
            sub.set_xlabel(f'ROI: {i}')
            temp = data_image[int(locs[i][1]):int(locs[i][1] + ROI_size[1]),
                              int(locs[i][0]):int(locs[i][0] + ROI_size[0])]
            sub.imshow(temp,
                       interpolation='bilinear',
                       cmap='autumn')
        plt.pause(1)
        plt.close(fig)
    else:
        plt.title('Area to be analysed', fontsize=16)
        pos = locs[0] if locs[1][0] > locs[0][0] else locs[1]
        size_x = max(locs[1][0], locs[0][0]) - min(locs[1][0], locs[0][0])
        size_y = max(locs[1][1], locs[0][1]) - min(locs[1][1], locs[0][1])
        rect = patches.Rectangle((pos[0], pos[1] - size_y),
                                 size_x,
                                 size_y,
                                 linewidth=2,
                                 edgecolor='w',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.pause(1)
        plt.close(fig)

    return locs
