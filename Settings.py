root_path = '/Volumes/krauss/Isabel'

exp_path = '/Phenospot_TF_data_science_support/Matlab/data/Location_1'

HyS_path = f'{root_path}/{exp_path}/Hyperspectral'
PhC_path = f'{root_path}/{exp_path}/Phasecontrast'
savepath = f'{root_path}/{exp_path}'
mapfilename = ''

wave_initial = 670
wave_final = 690
wave_step = 0.2

# How many hours after cell activation was the first image taken?
act_time = 0  # hrs

cell_num = 20

histogram_name = 'loc_01_ROI_av_hist.png'
res_over_t_name = 'res_curves_wav_t_loc_01_ROI_av.png'
boxplot_name = 'boxplot_loc_01_ROI_av.png'
videoname = 'loc01_ROI_av.mp4'
