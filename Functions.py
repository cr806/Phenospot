import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredText

FONT_PARAMS = {'figure.titlesize': 20,
               'axes.titlesize': 15,
               'axes.labelsize': 15,
               'xtick.labelsize': 12,
               'ytick.labelsize': 12,
               'legend.fontsize': 12}


def save_phasecontrast_video(time_annotation,
                             image_paths,
                             video_filename,
                             t_interval,
                             video_length):
    ''' Function to create a video (with annotations) from a series of images.
        Images are first flipped up for down, then normalised and finally
        filtered to remove salt and pepper noise
        Args:
            time_annotation: <list> List of times in hours
                             (e.g. [1, 2.1, 4.5, ...])
            image_paths: <list> List of filepaths pointing to images
    '''
    if (FPS := (len(t_interval) // video_length)) < 1:
        FPS = 1

    writer = FFMpegWriter(fps=FPS)

    PhC_image = np.array(Image.open(image_paths[0]))
    PhC_image = PhC_image - np.amin(PhC_image)
    PhC_image = PhC_image / np.amax(PhC_image)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), layout='tight')

    img_PhC = ax.imshow(PhC_image, aspect='auto', cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title('Phase contrast image')

    with writer.saving(fig, video_filename, dpi=100):
        for idx, (t, PhC_fp) in enumerate(zip(time_annotation, image_paths)):
            temp = np.array(Image.open(PhC_fp))
            temp = temp - np.amin(temp)
            temp = temp / np.amax(temp)

            img_PhC.set_data(temp)

            at = AnchoredText(f'Time: {t:0.1f} h',
                              prop=dict(size=20),
                              frameon=True,
                              loc='lower left')
            at.patch.set_boxstyle('square, pad=0.1')
            ax.add_artist(at)

            writer.grab_frame()
            print(f'Frame {idx + 1} of {len(image_paths)} written', end='\r')


def build_image_stack(images_path, image_size, img_start_idx, img_end_idx):
    ''' Function to create a 3D array of images
        Args:
            image_paths: <list> List of filepaths pointing to images
            image_size: <tuple> (int, int) Size of the images
            img_start_idx: <int> Index of the first image to be included
            img_end_idx: <int> Index of the last image to be included
        Returns:
            imstack: <list> List of 2D numpy arrays containing the image
                     data (i.e. brightness values from each pixel)
    '''
    data_paths = [h for h in Path(images_path).glob('*.tiff')]
    data_paths.sort(key=lambda x: int(x.stem.split('_')[1]))
    data_paths = data_paths[img_start_idx:img_end_idx + 1]

    imstack = np.zeros((image_size[1], image_size[0], len(data_paths)))
    print(f'Creating image stack size: {imstack.shape}')
    for idx, d in enumerate(data_paths):
        with Image.open(d) as im:
            imstack[:, :, idx] = im
            print(f'Imported image {idx+1} of {len(data_paths)}', end='\r')

    return imstack


def get_resonance_idxs(imstack, method):
    ''' Function to return the locations within the imstack (axis=2) of the
        resonant pixel
        Args:
            imstack: <list> List of 2D numpy arrays containing the image
                     data (i.e. brightness values from each pixel)
            res: <float> Initial guess for resonant value
            res_width: <float> Initial guess for resonance FWHM
        Returns:
            2D numpy array, each entry being the index of the resonant pixel
            from the imstack along axis=2 (i.e. the time data)
    '''
    if method == 'max' or method == 'min':
        return use_peak(imstack, method)
    elif method == 'fano':
        return use_fano(imstack)


def fano(x, amp, assym, res, gamma, off):
    ''' Fano function used for curve-fitting

        Attributes:
        x <float> :    Independant data value (i.e. x-value, in this
                        case pixel number)
        amp <float>:   Amplitude
        assym <float>: Assymetry
        res <float>:   Resonance
        gamma <float>: Gamma
        off <float>:   Offset (i.e. function bias away from zero)

        Returns:
        float: Dependant data value (i.e. y-value)
    '''
    num = ((assym * gamma) + (x - res)) * ((assym * gamma) + (x - res))
    den = (gamma * gamma) + ((x - res)*(x - res))
    return (amp * (num / den)) + off


def use_fano(imstack):
    ''' Function to return index of resonant value within the imstack
        array along axis=2.  Fits Fano poynomial to data to return 'true'
        resonant location.
        Args:
            imstack: <list> List of 2D numpy arrays containing the image
                     data (i.e. brightness values from each pixel)
            res: <float> Initial guess for resonant value
            res_width: <float> Initial guess for resonance FWHM
        Returns:
            2D numpy array, each entry being the index of the resonant pixel
            from the imstack along axis=2 (i.e. the time data)
    '''

    # Use maximum pixel location as initial resonance guess
    res_array = np.argmax(imstack, axis=2)
    off_array = np.mean(imstack, axis=2)
    amp_array = np.amax(imstack, axis=2) - off_array
    xdata = np.arange(0, imstack.shape[2], 1)
    results = np.zeros(imstack[:, :, 0].shape)
    res_width = 50

    for i in range(imstack.shape[0]):
        for j in range(imstack.shape[1]):
            ydata = imstack[i, j, :]
            res = res_array[i, j]
            off = off_array[i, j]
            amp = amp_array[i, j]

            initial = [amp, 0, res_width, res, off]
            bounds = ([(0.5 * amp), -1, (0.90 * res_width),
                       (0.95 * res), (0.5 * off)],
                      [(1.5 * amp),  1, (1.10 * res_width),
                       (1.05 * res), (1.5 * off)])

            try:
                popt, _ = curve_fit(fano, xdata, ydata,
                                    p0=initial, bounds=bounds)
                _, _, _, res_pos, _ = popt
            except (RuntimeError, ValueError):
                res_pos = 0
            results[i, j] = res_pos
    return results


def use_peak(imstack, method):
    ''' Simple function to return index of maximum value within the imstack
        array along axis=2.
        Args:
            imstack: <list> List of 2D numpy arrays containing the image
                     data (i.e. brightness values from each pixel)
        Returns:
            2D numpy array, each entry being the index of the resonant pixel
            from the imstack along axis=2 (i.e. the time data)
    '''
    if method == 'max':
        return np.argmax(imstack, axis=2)
    elif method == 'min':
        return np.argmin(imstack, axis=2)


def get_area(data_image, image=None):
    ''' Function to return two (x,y) coordinates as chosen by the user
        Args:
            data_image: 2D numpy array containing the image data
        Returns:
            2D numpy array, each entry being the index of the resonant pixel
            from the imstack along axis=2 (i.e. the wavelength data)
    '''
    if not image:
        fig, ax = plt.subplots(1, 1)
        plt.suptitle('Area to be analysed', fontsize=16)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    else:
        ax = image
    img = ax.imshow(data_image,
                    interpolation='bilinear',
                    cmap='autumn')
    # img.set_clim(vmin=Settings.wave_initial,
    #              vmax=Settings.wave_final - Settings.wave_step)
    if not plt.bar:
        bar = plt.colorbar(img)
        bar.set_label('Resonant Wavelength [nm]')

    temp = list()
    while True:
        plt.title(f'You have selected {len(temp)} / 2 points of the ROI.',
                  fontsize=16)
        plt.draw()
        if len(temp) >= 2:
            break
        pt = plt.ginput(1, timeout=-1)[0]
        temp.append((int(pt[0]), int(pt[1])))

    locs = list()
    locs.append((min(temp[0][0], temp[1][0]), min(temp[0][1], temp[1][1])))
    locs.append((max(temp[0][0], temp[1][0]), max(temp[0][1], temp[1][1])))

    pos = locs[0] if locs[1][0] > locs[0][0] else locs[1]

    size_x = max(locs[1][0], locs[0][0]) - min(locs[1][0], locs[0][0])
    size_y = max(locs[1][1], locs[0][1]) - min(locs[1][1], locs[0][1])
    rect = patches.Rectangle(pos,
                             size_x,
                             size_y,
                             linewidth=2,
                             edgecolor='w',
                             facecolor='none')
    ax.add_patch(rect)
    if not image:
        plt.pause(1)
        plt.close(fig)
        return locs
    else:
        return locs, ax


def get_ROIs(data_image, num_of_pts=4, ROI_size=(100, 100)):
    ''' Function to return multiple (x,y) coordinates as chosen by the user
        Args:
            daya_image: 2D numpy array containing the image data
            num_of_pts: <int> Number of coordinates to request from user
            ROI_size: <tuple> (int, int) x- and y-size of region or interest,
                      used to correct coordinate to lower left of ROI box and
                      to disply ROI to user overlaid on image
        Returns:
            List of 2D numpy arrays, each entry being the x and y coordinate
            of the chosen ROI, adjusted for the size of the ROI
    '''
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
        pt = (pt[0] - ROI_size[0]/2, pt[1] - ROI_size[1]/2)
        locs.append((int(pt[0]), int(pt[1])))

    plt.pause(0.5)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))  # width, height in inches
    plt.title('ROIs to be measured', fontsize=16)
    plt.gca().axes.axis('off')
    sub_x = int(np.ceil(np.sqrt(num_of_pts)))
    sub_y = int(np.ceil(num_of_pts / sub_x))
    for i in range(num_of_pts):
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

    return locs


def get_ROI_areas(data_image, num_of_pts=4):
    ''' Function to return multiple [(x,y)(x,y)] coordinates as chosen by the
        user defining the corners of the ROIs
        Args:
            data_image: 2D numpy array containing the image data
            num_of_pts: <int> Number of coordinates to request from user
        Returns:
            List of 2D numpy arrays, each entry containing the x and y
            coordinate of the bottom left and top right of the chosen ROI
    '''
    fig, ax = plt.subplots(1, 1)
    plt.suptitle(f'Select {num_of_pts} ROIs', fontsize=16)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    locs = list()
    for _ in range(num_of_pts):
        (pt, ax) = get_area(data_image, image=ax)
        locs.append(pt)
    plt.pause(1)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))  # width, height in inches
    plt.title('ROIs to be measured', fontsize=16)
    plt.gca().axes.axis('off')
    sub_x = int(np.ceil(np.sqrt(num_of_pts)))
    sub_y = int(np.ceil(num_of_pts / sub_x))
    for i in range(num_of_pts):
        sub = fig.add_subplot(sub_x, sub_y, i + 1)
        sub.axes.get_xaxis().set_ticks([])
        sub.axes.get_yaxis().set_ticks([])
        sub.set_xlabel(f'ROI: {i}')
        temp = data_image[locs[i][0][1]:locs[i][1][1],
                          locs[i][0][0]:locs[i][1][0]]
        sub.imshow(temp,
                   interpolation='bilinear',
                   cmap='autumn')
    plt.pause(1)
    plt.close(fig)
    return locs
