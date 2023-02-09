'''########################################################################'''
'''#                 General experimental settings                        #'''
'''########################################################################'''
root_path = '/Users/chris/Documents/SoftwareDev/Python'
exp_path = 'Phenospot/Data/Location_1'

'''########################################################################'''
'''#                     Resonant map Settings                            #'''
'''########################################################################'''
wave_initial = 670
wave_final = 674.2  # 690
wave_step = 0.2

peak = True

map_data = 'mapstore.npy'

'''########################################################################'''
'''#                 Phase-contrast video settings                        #'''
'''########################################################################'''
# How many hours after cell activation was the first image taken?
act_time = 0  # hrs

PhC_video = 'loc01_ROI_av.mp4'

'''########################################################################'''
'''#                    Data analysis settings                            #'''
'''########################################################################'''
cell_num = 2

roirect_data = 'roi_rect.npy'
av_shift_data = 'av_shift.npy'
histogram_data = 'loc_01_hist_ROI_av.npy'
histogram_chart = 'loc_01_ROI_av_hist.png'
res_over_t_chart = 'res_curves_wav_t_loc_01_ROI_av.png'
boxplot_chart = 'boxplot_loc_01_ROI_av.png'
HyS_video = 'loc01_ROI_HyS.mp4'
