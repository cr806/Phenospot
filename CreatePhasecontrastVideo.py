from pathlib import Path
from datetime import datetime

import Config as cfg
import Functions as fn


def create_phase_contrast_video():
    PhC_path = Path(cfg.root_path, cfg.exp_path, 'Phasecontrast')

    PhC_data_paths = [p for p in PhC_path.glob('*.tiff')]
    PhC_data_paths.sort(key=lambda x: x.parts[-1])

    mod_time = list()
    for PhC_fp in PhC_data_paths:
        m_timestamp = PhC_fp.stat().st_mtime
        mod_time.append(datetime.fromtimestamp(m_timestamp))

    t_interval = [((t - mod_time[0]).seconds/3600) +
                  cfg.act_time for t in mod_time]

    save_path = Path(cfg.root_path, cfg.exp_path, cfg.PhC_video)
    fn.save_phasecontrast_video(t_interval, PhC_data_paths, save_path)

    return t_interval


if __name__ == '__main__':
    _ = create_phase_contrast_video()
