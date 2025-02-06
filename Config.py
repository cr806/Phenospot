########################################################################
#                 General experimental settings                        #
########################################################################
root_path = '/Volumes/krauss/Lisa/Phenospot/Biofilms'
expt_path = 'Experiment 1 - Test run 250121/2_biofilm_18h'

########################################################################
#                     Experiment Settings                            #
########################################################################
wave_initial = 622
wave_final = 632
wave_step = 0.2

image_size = (1920, 1460)

########################################################################
#                     Resonant map Settings                            #
########################################################################
wave_slice_start = 622
wave_slice_end = 632

method = 'max'
# method = 'min'
# method = 'fano'

HyS_image_filename = 'Hyperspectral_Image.png'

########################################################################
#                 Phase-contrast video settings                        #
########################################################################

PhC_video_filename = 'PhaseContrastVideo.mp4'
video_length = 20
