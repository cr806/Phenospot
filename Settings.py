root_path = '/Volumes/krauss/Isabel'

exp_path = '/Phenospot_TF_data_science_support/Matlab/data/Location_1'

HyS_path = f'{root_path}/{exp_path}/Hyperspectral'
PhC_path = f'{root_path}/{exp_path}/Phasecontrast'
savepath = f'{root_path}/{exp_path}'


wave_initial = 670
wave_final = 690
wave_step = 0.2

ROI_size = (100, 100)

peak = True

# How many hours after cell activation was the first image taken?
act_time = 0  # hrs

cell_num = 20

mapfilename = 'test_mapstore.npy'
roirectfilename = 'test_roirect.npy'
av_shiftfilename = 'test_avshift.npy'
histogram_workspace = 'loc_01_hist_ROI_av.npy'
histogram_name = 'loc_01_ROI_av_hist.png'
res_over_t_name = 'res_curves_wav_t_loc_01_ROI_av.png'
boxplot_name = 'boxplot_loc_01_ROI_av.png'
videoname = 'loc01_ROI_av.mp4'
HyS_videoname = 'loc01_ROI_HyS.mp4'
