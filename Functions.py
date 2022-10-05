import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredText

import Settings


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

    # np.save(f'imstack_{H_idx}.npy', imstack)  # Results in a 2.2Gb file, larger than images when stored separately

    return imstack
