import numpy as np
from pathlib import Path
from PIL import Image

import Config as cfg
import Functions as fn


def create_res_map():
    HyS_path = Path(cfg.root_path, cfg.exp_path, 'Hyperspectral')

    HyS_data_paths = [p for p in HyS_path.iterdir() if p.is_dir()]
    HyS_data_paths.sort(key=lambda x: int(x.parts[-1]))

    temp_path = list(HyS_data_paths[0].glob('*.tiff'))[0]
    temp_im = Image.open(temp_path)

    map_store = np.zeros((temp_im.size[1],
                         temp_im.size[0],
                         len(HyS_data_paths)))

    # Why do we remove a wave_step from wave_final
    wav_ref = np.arange(cfg.wave_initial,
                        cfg.wave_final - cfg.wave_step,
                        cfg.wave_step)

    im_list = list()
    for H_idx, HyS_fp in enumerate(HyS_data_paths):

        imstack = fn.build_image_stack(HyS_fp)
        # Results in a 2.2Gb file, larger than images when stored separately
        # np.save(f'imstack_{H_idx}.npy', imstack)

        resonance_indexes = fn.get_resonance_idxs(imstack)
        map_store[:, :, H_idx] = wav_ref[resonance_indexes]

        # Multiply resonant data by 10 so not to lose resolution (i.e. take
        # one decimal place f64 and make compatible with int16) then scale for
        # uint16 (i.e. min value 0, max 65536)
        u16in = (((wav_ref[resonance_indexes] - cfg.wave_initial) * 10)
                 * (65536 / ((cfg.wave_final - cfg.wave_initial) * 10)))
        u16in = u16in.astype(np.uint16)
        out_pil = u16in.astype(u16in.dtype.newbyteorder('<')).tobytes()
        img_out = Image.frombytes('I;16', (1920, 1460), out_pil)
        im_list.append(img_out)

        print(f'Resonant map {H_idx + 1} of {len(HyS_data_paths)} complete')

    save_path = Path(cfg.root_path, cfg.exp_path, cfg.map_data)
    print(f'Resonant map data -> {str(save_path)}')
    np.save(save_path, map_store)

    save_path = Path(cfg.root_path,
                     cfg.exp_path,
                     f'{cfg.map_data[:-4]}.tiff')
    print(f'Resonant map data as TIFF stack -> {str(save_path)}')
    im_list[0].save(save_path,
                    format='TIFF',
                    save_all=True,
                    append_images=im_list[1:])

    return map_store


if __name__ == '__main__':
    map_store = create_res_map()
