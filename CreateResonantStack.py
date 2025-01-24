import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from Config import (root_path, expt_path, wave_initial,
                    wave_final, wave_step, image_size,
                    method, HyS_image_filename)
from Functions import build_image_stack, get_resonance_idxs


def create_res_maps(break_at=-1):
    try:
        dtypes = {
            'Filepath': 'string',
            'Processed': 'bool',
            'Result': 'string'
        }
        HyS_df = pd.read_csv(Path(root_path, expt_path, 'HyS_results.csv'),
                             dtype=dtypes)
        print('HyS_results.csv found and loaded.')
    except FileNotFoundError:
        root_len = len(Path(root_path).parts)
        HyS_path = Path(root_path, expt_path, 'Hyperspectral')

        HyS_data_paths = [Path(*p.parts[root_len:])
                          for p in HyS_path.iterdir()
                          if p.is_dir()]
        HyS_data_paths.sort(key=lambda x: int(x.stem))
        HyS_df = pd.DataFrame(HyS_data_paths,
                              columns=['Filepath'],
                              dtype='string')
        HyS_df['Processed'] = False
        HyS_df['Processed'] = HyS_df['Processed'].astype(bool)
        HyS_df['Result'] = 'None'
        HyS_df['Result'] = HyS_df['Result'].astype('string')
        HyS_df.to_csv(Path(root_path, expt_path, 'HyS_results.csv'),
                      index=False)
        print('HyS_results.csv not found. Creating new file.')

    HyS_data_paths = HyS_df[HyS_df['Processed'] == False]['Filepath']  # noqa: E712, E501

    if HyS_data_paths.empty:
        print('All HyS data has been processed. Exiting...')
        return

    wav_ref = np.arange(wave_initial,
                        wave_final + wave_step/2,
                        wave_step)

    assert len(wav_ref) == len(list(Path(HyS_data_paths.values[0]).glob(
            '*.tiff'))), 'Wavelength array must the number of HyS images'

    for idx, fp in enumerate(HyS_data_paths):
        if break_at > 0 and idx == break_at:
            break
        print(f'Processing "{fp}"')
        imstack = build_image_stack(Path(root_path, fp), image_size)

        resonance_indexes = get_resonance_idxs(imstack, method)
        HyS_image = wav_ref[resonance_indexes]

        # Multiply resonant data by 10 so not to lose resolution (i.e. take
        # one decimal place f64 and make compatible with int16 (0 -> 65536))
        u16in = HyS_image * 10
        u16in = u16in.astype(np.uint16)
        out_pil = u16in.astype(u16in.dtype.newbyteorder('<')).tobytes()
        img_out = Image.frombytes('I;16', image_size, out_pil)
        img_out.save(Path(root_path, fp, HyS_image_filename))

        # Update dataframe with resulting filename and set processed to True
        HyS_df.loc[HyS_df['Filepath'] == fp, 'Processed'] = True
        HyS_df.loc[HyS_df['Filepath'] == fp, 'Result'] = str(
                                                 Path(expt_path,
                                                      HyS_image_filename))
        # Update progress file on disk every 10th iteration
        if idx % 10 == 0:
            HyS_df.to_csv(Path(root_path, expt_path, 'HyS_results.csv'),
                          index=False)

    HyS_df.to_csv(Path(root_path, expt_path, 'HyS_results.csv'), index=False)


if __name__ == '__main__':
    create_res_maps(break_at=2)
