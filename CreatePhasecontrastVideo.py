from pathlib import Path

from Config import root_path, expt_path, video_length, PhC_video_filename
from Functions import save_phasecontrast_video


def create_phase_contrast_video():
    PhC_path = Path(root_path, expt_path, 'Phasecontrast')

    PhC_data_paths = [p for p in PhC_path.glob('*.tiff')]
    PhC_data_paths.sort(key=lambda x: int((x.name).split('_')[0]))

    mod_time = list()
    for PhC_fp in PhC_data_paths:
        m_timestamp = PhC_fp.stat().st_mtime
        mod_time.append(m_timestamp)

    t_interval = [((t - mod_time[0])/3600) for t in mod_time]

    save_path = Path(root_path, expt_path, PhC_video_filename)
    if save_path.exists():
        print('A Phasecontrast video already exists. Skipping creation.')
        return t_interval

    save_phasecontrast_video(t_interval,
                             PhC_data_paths,
                             save_path,
                             t_interval,
                             video_length)

    return t_interval


if __name__ == '__main__':
    _ = create_phase_contrast_video()
