%% Image processing script

%% Clear all variables and figures pre-run
clear all
close all

%% INPUT: if you want to include a loop to pre-read the next location, set the number of locations here or set to 1
number_of_wells = 1;

for analyse_all_wells = 1:number_of_wells

%% Initialisations
% Copy directory of location where the wavelength sweeps are saved in
% folder 0 1 ... don't forget backslash

if analyse_all_wells == 1
    clear all
    close all
    HyS_path = 'D:\Alasdair\Phenospot data\2024-10-03\Main Expt\Location_1\HyperSpectral\'; % this is where the hyperspectral data is saved in the folders 1,2,3,...
    root = HyS_path;
    PhC_path = 'D:\Alasdair\Phenospot data\2024-10-03\Main Expt\Location_1\PhaseContrast\'; %This path has to be where the phase contrast images were saved
    savepath = 'D:\Alasdair\Phenospot data\2024-10-03\Main Expt\Location_1\Output\'; %This can be any path where you want to save the final histogram workspaces for further processing, not the 2 above
    scriptpath = 'P:\Phenospot Data\Sensor Tests\Drift testing\scripts\';
    analyse_all_wells = 1;
    number_of_wells = 1;
end

%% Enable the following code to automatically move onto a second location processing, although it will only progress to ROI selection 
% if analyse_all_wells == 2
    % clear all
    % close all
    % HyS_path = 'R:\rsrch\tf-IC\lab\I&C\Phenospot\Data analysis test folder\Test location 2\Hyperspectral\'; % this is where the hyperspectral data is saved in the folders 1,2,3,...
    % root = HyS_path;
    % PhC_path = 'R:\rsrch\tf-IC\lab\I&C\Phenospot\Data analysis test folder\Test location 2\PhaseContrast\'; %This path has to be where the phase contrast images were saved
    % savepath = 'R:\rsrch\tf-IC\lab\I&C\Phenospot\Data analysis test folder\Test location 2\Output\'; %This can be any path where you want to save the final histogram workspaces for further processing, not the 2 above
    % scriptpath = 'P:\Phenospot Data\Macrophage Project\Alasdair\2023-05-23\Scripts\';
    % analyse_all_wells = 2;
    % number_of_wells = 2;
% end
%% Disable above code for single location processing


%% Define name of file where all hyperspectral maps are stored and saved
mapfilename = 'mapstore_loc1.mat';

%% INPUT: set this variable to y if you want all csv to be output for each cell location at each timepoint and all median figures
OutputAllData = 'n';

%% INPUT: define wavelength sweep range
%% Location 1
wav_ini = 625.0;
wav_fin = 635.2;

%% Enable following codes for multiple locations that use different wavelengths but add an if loop based on analyse_all_wells number
%% Location 2
% wav_ini = 667;
% wav_fin = 675.2;

%% INPUT: set the steps between each wavelength in the images taken
% step = 0.5;
step = 0.2;
% step = 1.0;

%% INPUT: Define size of area to be selected around the cell
%% -64 and 128 is a square 128 x 128 with cell in centre
%% Used for 10X imaging
lolimit = -64;
hilimit = 128;
%% Large region for non-cell specific selection
% lolimit = -300;
% hilimit = 600;
%% -32 and 64 is a square 64 x 64 with cell in centre
%% Used for 4X imaging
% lolimit = -32;
% hilimit = 64;
%% Used for 4X imaging
% lolimit = -20;
% hilimit = 40;



%% Define where the peak is roughly located by indicating f1 = 1/2 if in the middle f1=1/4 or 3/4 if in first or second half of range
%% and give approximate FWHM in percetange of the whole range, e.g. 1/2 or 1/4
%% This code is rarely changed as it doesn't affect processing in any major way
%% Graph axes have been modified (AK) to use min and max rather than centre around the peak
fr1 = 0.75;
fr2 = 0.25;
pos_initial = wav_ini+(wav_fin-wav_ini)*fr1;
peak_width = abs(wav_fin-wav_ini)*fr2;
%%
%% INPUT: Select the number of cells to be examined as a default
%% This is over-ridden if the user enters a cell number based on the images but default if they put 0 (Line 522-531 ish)
% cellnum = 40; % edit to 5 for speed testing
% cellnum = 8;
cellnum = 5;
%%

%% INPUT: How many hours after cell activation was the first image taken? Use zero if activated at experiment start
act_time = 0; 

%% Check if the output folder exists and create it if not
if not(isfolder(savepath))
        mkdir(savepath)
end

%% Set a folder name that varies each run to allow unique files for repeat analyses
folderdate = datetime('now');
outputFolder = ['Output ' [num2str(folderdate.Year)] ' ' [num2str(folderdate.Month)] ' ' [num2str(folderdate.Day)] ' ' [num2str(folderdate.Hour)] 'hr ' [num2str(folderdate.Minute)] 'mins'];
    outSave = [savepath outputFolder];

%% Check if the above folder already exists and create it if not
    if not(isfolder([savepath outputFolder]))
        mkdir([savepath outputFolder])
    end

savepath = [outSave '/'];

%% add output of selected cells image
selected_cells_name = 'map_of_selected_cells.png';

%% set up names for saved files
av_reswav_name = 'av_res_whole_FOV_01_fin10.png';
av_reswav_workspace = 'av_res_whole_FOV_01_fin10.mat';
    videoname = 'loc01_zoom_fin10';
     videoname_hyper = 'loc01_hyper_zoom_fin10';
    framerate = 2; %frames per second
    Fontsize_video = 50;   
     % Fontsize_video = 100;

    reswavname = 'avres_well1_antibodies.png';
    time_table_name = 'time.csv';
    reswav_table_name = 'reswav_01.csv';
    sigma_table_name = 'sigma_01.csv';% standard deviation of averaged resonance wavelength in ROI  
   
    histogram_name = 'loc_01_hist_fin10.png';
    res_over_t_name = 'res_curves_wav_t_loc_01_fin10.png';
    boxplot_name = 'boxplot_loc_01_fin10';
    boxplot_cell_means = 'boxplot_cell_means';
    boxplot_total_mean = 'boxplot_total_mean';
    boxplot_cell_medians = 'boxplot_cell_medians';
    boxplot_total_median = 'boxplot_total_median';
    plot_total_mean = 'plot_all_cells_mean';
    histogram_time_name = 'boxplot histogram over time.png';
    hyp_over_time_name = 'hyperspec over time';
    hyp_mean_name = 'hyperspec mean';
    hyp_mean_no_norm_name = 'hyperspec mean no norm';
    hyp_median_name = 'hyperspec median';
    hyp_all_norm_name = 'hyperspec mean norm to ref and zero time';
    histogram_workspace = 'loc_01_hist_fin10.mat'; 
    legend_name = 'well 01 - test ';
   
    %%
cd(PhC_path)

data = dir('*.tiff');
filenames = fullfile( {data.folder}, {data.name});
cd(scriptpath)
filenames = natsort(filenames);
cd(PhC_path)
%% set nfiles to be the number of timepoints
nfiles = length(filenames);

%%
%% INPUT: Choose all recorded scans (nfiles) or specific range by setting fin to a specific number
ini = 1;
fin = (nfiles);
fin_hist = (nfiles);
%% Set tally to enable matlab to tell you which timepoint you are importing
Tally = 0;

filename = filenames{1};
all_data_cell = cell(nfiles, 1);
basenames = cell(nfiles, 1);
all_data_cell{1} = imread(filename);
       
A = (all_data_cell{1});
%%
cd(HyS_path)
D = dir;
cd(root)
reswav = zeros(1,length(D)-2);
timeelapsed = 0:1:(1*length(D)-2);
mapstore = zeros(1460,1920,length(D)-2);


%%
cd(PhC_path)

for i = ini:fin
    try
        info = imfinfo(filenames{i});
        timestamp{i} = datevec(info.FileModDate);
        t_interval(i) = (etime(timestamp{i},timestamp{1}))./60/60 + act_time; %in hours
    catch ME
        fprintf('Warning: problem retrieving image information')
    end
end

%% Look at phase contrast images and save video --> visual check #1
    for K = ini:fin
      filename = filenames{K};
      [~, basenames{K}, ~] = fileparts(filename);
       try
        all_data_cell{K} = imread(filename);
       catch ME
        fprintf('Warning: problem importing file "%s"\n', filename);
      end
      K
    end
%%
for d = ini:fin
    
    clear avspecint 
    timemultiple = d;
    cd(scriptpath)
    wavref = wav_ini:step:wav_fin-step;
    wavint = interp1(1:length(wavref),wavref,linspace(1,length(wavref),1000));
    wavref = interp1(1:length(wavint),wavint,linspace(1,length(wavint),length(wavref)));
    peak = 1;
    colaxis = [wav_ini wav_fin];
%% Data importing

Tally = Tally + 1;
format = ['Importing timepoint %d now \n'];
fprintf(format,Tally);

% Change directory to where images are saved
% read in each image as an individual variable: im1,im2,etc.
folstring = strcat(HyS_path,num2str(timemultiple),'/');
cd(folstring)
format = 'Image %d of %d imported \n';
A = dir('*.TIFF');

C = {A.name};
cd(scriptpath)
Aname = natsort(C);
cd(folstring)

%% CR code

        temp = importdata(string(Aname(1)));
        xsize = size(temp, 1);
        ysize = size(temp, 2);
        imstack = zeros(xsize, ysize, length(A));
        clear temp

        for n = 1:length(A)
            temp = importdata(string(Aname(n)));
            imstack(:, :, n) = temp(:, :, 1);
            fprintf(format,n,length(A));
            clear temp
        end
        clear n

        format = 'Data importing finished \n';
        fprintf(format);

%% end of CR code


%% Main loop to locate resonance at every pixel

if peak == 1   						
    [~,I] = max(imstack,[],3);
else
    [~,I] = min(imstack,[],3);
end
map = wavref(I);
mapstore(:,:,d) = map;
clear I

format = 'Map finished \n';
fprintf(format);

 %% Set the colourbar to the min and max of what is on this plot
 if pos_initial < wav_ini
     caxis_left = pos_initial - wav_ini;
     caxis_right = wav_fin - pos_initial;
 else
     caxis_left = pos_initial - wav_ini - 2;
     caxis_right = wav_fin - pos_initial - 2;
 end

%% altered to catch if there is no signal i.e. pos_initial is less than wav_ini

 
%% Display Figure 1 the resonance map from which the ROI is selected
fig1 = figure(1);
pcolor(wiener2(map,[2 2]))
shading interp
caxis([colaxis(1) colaxis(2)]);
%% to save it here use saveas(gca, 'Full ROI selected.png') - this code doesn't function correctly but is not currently required
% saveas(gca, [savepath 'Full ROI selected.png'])
colormap(jet);
colorbar
title('Resonance map','FontSize',12)
axis equal
set(gca,'xticklabel','','yticklabel','')
format = 'Plotting finished \n';
fprintf(format);

%% Select total ROI that covers the sensor
if d == 1
    format = 'Select ROI for average resonance check clicking bottom left corner to top right corner \n';
    fprintf(format);
    [x,y] = ginput(2);
end

x = round(x); y = round(y);

region = imstack(y(1):y(2),x(1):x(2),1:end);

%% clear imstack to save memory
clear imstack;

mapregion = map(y(1):y(2),x(1):x(2));

%% store all regions to assess later
hypstore(:,:,d) = mapregion;

reswav_std(d) = std(std(mapregion));

ar = [x(1) x(2);y(2) y(2)];
regsize = size(region);
close(fig1)
%%

%% ROI interpolation
temp = reshape(region,regsize(1)*regsize(2),length(wavref));
avspec = mean(temp,1);
clear temp


%% Average spectrum calculation, levelling, and plotting from interpolation
avspecint = interp1(wavref,avspec,wavint);
levellingint = linspace(avspecint(1),avspecint(end),length(avspecint));
cd(scriptpath)
[specfit,gof] = fanofit(wavint,avspecint,peak_width,pos_initial);
fitted = feval(specfit,wavint);

%% Average spectrum analysis
[~,in] = max(fitted);
reswav(d) = wavint(in);
%%

if reswav(1) > wav_ini
    caxis_left = reswav(1) - wav_ini - 2;
    caxis_right = wav_fin - reswav(1) - 2;
else
    caxis_left = reswav(1) - wav_ini;
    caxis_right = 2;
end
%% adjusted to catch when reswav and wav_ini are the same i.e. no signal

%% Figure 2 shows the selected ROI on the hyperspectral and phase contrast for each timepoint in turn, data output as video
fig2 = figure(2);
fig2.Position(1:4) = [100 200 1600 500];
subplot(1,2,1)
    pcolor(wiener2(wiener2(mapregion + abs((mean(mean(mapregion(:,:,1))) - mean(mean(mapregion)))),[10 10])))
    shading interp
    caxis([colaxis(1) colaxis(2)]);
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Resonance wavelength \lambda [nm]';
    title('Resonance map','FontSize',12)
    set(gca,'xticklabel','','yticklabel','')
    set(gca,'Fontsize',15)
    axis tight equal

subplot(1,2,2)
    A = (all_data_cell{d});
    A = A(y(1):y(2),x(1):x(2));
    A = flipud(medfilt2(medfilt2((imadjust(medfilt2(A(:,:,1)))))));

%% store the phase contrast images sequentially in "running cell" so that the selected cells can be displayed later
%% this enables tracking of cells within the selected region to make sure they are present throughout the experiment
runningCell{:,:,d} = flipud(A);

    text_str = ['Time:' num2str(t_interval(d),'%0.1f') ' h'];
    position = [10 10]; 
    box_color = {'blue'};

    RGB = insertText(A,position,text_str,'FontSize',Fontsize_video,'BoxColor',box_color,'BoxOpacity',0.1,'TextColor','white');

    imshow(RGB)
    title('Cell Images','FontSize',12)
    set(gca,'xticklabel','','yticklabel','')
    set(gca,'Fontsize',15)
    axis tight equal
    F2(d-ini+1) = getframe(fig2);    
        
 
%% Figure 3 is the line graph for peak resonance, determined as "max" resonance signal in the ROI for each timepoint
%% each graph has r2 for fit and is output as a png file to be examined after
fig3 = figure(3)
    text_2 = ['Time:' num2str(t_interval(d),'%0.1f') ' h   ' 'r-squared:' num2str(gof.rsquare) ' '];
    p = plot(wavref,avspec,'or',wavint,fitted,'-k');
    set(p,'MarkerSize',2,'LineWidth',1);
    xlabel(['Wavelength (nm)      ' text_2] )
    ylabel('Reflectance [a.u.]')
    set(gca,'Fontsize',15)
    
    saveas(fig3,[savepath 'Full ROI peak graph timepoint ' num2str(d) '.png']);

format = 'Average spectrum calculation finished \n';
fprintf(format);

end
%%

cd(savepath)

%% make video from Fig 2 with hyperspectral and phase contrast images at each timepoint
v2 = VideoWriter(videoname_hyper,'MPEG-4'); 
v2.FrameRate = framerate;
open(v2)
writeVideo(v2,F2)
close(v2)

%% Saving
save(av_reswav_workspace,'reswav')  
save(av_reswav_name,'reswav')
save(reswavname,'reswav')

%% Create a table with the data and variable names
t_table = table(t_interval);
reswav_table = table(reswav);
sigma_table = table(reswav_std);

%% Write data to text file
writetable(t_table, time_table_name)
writetable(reswav_table, reswav_table_name)
writetable(sigma_table, sigma_table_name)

%% ask user for how many cells they want to analyse
%% catch if they accidentally hit enter without putting a number and default to preset cellnum from start
prompt = "Please enter number of cells to be analysed ";
numberOfCells = input(prompt)

while isempty(numberOfCells)
    prompt = "Please enter number of cells to be analysed ";
    numberOfCells = input(prompt)
end

if numberOfCells < 1
    numberOfCells = cellnum
else
    cellnum = numberOfCells
end

%% select your map by selecting each individual cell
format = 'Select ROIs manually \n';
fprintf(format);

%% n loop for each cell selection
for n = 1:cellnum
%% this is where ROI is selected on phase contrast
    format = 'Running my added code next \n';
    fprintf(format);

%% select cells area from the cropped image selected at the start
    
    flippedA = flipud(A);
    fig21 = figure(21);
    fig21, imshow(flippedA);

    cellspot = images.roi.Point;
    pickedcell = drawpoint;
    wait(pickedcell);
    cellPosition = pickedcell.Position;
    xcoord = cellPosition(1, 1);
    ycoord = cellPosition(1, 2);

%% use lolimit and hilimit variables set at the start of the code to define area size 
    xlo = xcoord + lolimit;
    xhi = hilimit;
    ylo = ycoord + lolimit;
    yhi = hilimit;
    point_coord_x(n) = xcoord;
    point_coord_y(n) = ycoord;
    cellMarker{n} = num2str(n);
    PhaseCell = imcrop(flippedA,[xlo,ylo,xhi,yhi]);

%% added section here to display hyperspectral match to ensure the area selected is correct and then to allow area resizing if needed
%% Figure 4 shows whole hyperspectral of ROI with selected cell locations marked, and the region surrounding the cell point picked

    fig4 = figure(4);
    fig4.Position(1:4) = [200 200 800 500]; %adjusted location
            
    subplot(1,2,1) 
        imshow(mapregion)
        if n > 1
            for q = 1:(n-1)
                h = drawpoint('Position',[point_coord_x(q) point_coord_y(q)],'Label',cellMarker{q},'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='k');
            end
        end
        hold on;
        rectangle('Position', [xlo ylo xhi yhi] )
        hold on;
        shading interp
        caxis([colaxis(1) colaxis(2)]);
        colormap(jet)
        c = colorbar;
        c.Label.String = 'Resonance wavelength \lambda [nm]';
% 
    subplot(1,2,2) 
        [HShighlights{n}, location{n}] = imcrop(mapregion,[xlo,ylo,xhi,yhi]); %this is the working version 
            
        imshow(HShighlights{n})
        shading interp
        caxis([colaxis(1) colaxis(2)]);
        colormap(jet);
   
%% Figure 5 shows the phase contrast of just the cell region to be analysed based on the point click previously
%% The region can be adjusted if a second cell is visible or if the region needs to be resized
fig5 = figure(5);
    fig5.Position(1:4) = [600 100 800 500]; 
        [Phase_cells{n}, rect{n}] = imcrop(medfilt2((rescale(PhaseCell))));
        fig5, imshow(Phase_cells{n});

%% output phase cells and matching hyperspectral being assessed based on any resizing done on figure 5 - i.e. the final cell region for analysis
    fig6 = figure(6);
        fig6.Position(1:4) = [600 100 500 500];
        subplot(1,1,1)
        imshow(Phase_cells{n});
            
        if n == cellnum
            fig4 = figure(4);
            fig4.Position(1:4) = [200 200 800 500]; 
            
            subplot(1,2,1) 
                imshow(mapregion)
                for q = 1:cellnum
                    h = drawpoint('Position',[point_coord_x(q) point_coord_y(q)],'Label',cellMarker{q},'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='k');
                end
                hold on;
                shading interp
                caxis([colaxis(1) colaxis(2)]);
                colormap(jet)
                c = colorbar;
                c.Label.String = 'Resonance wavelength \lambda [nm]';
% 
            subplot(1,2,2)
                imshow(flippedA)
                for q = 1:cellnum
                    h = drawpoint('Position',[point_coord_x(q) point_coord_y(q)],'Label',cellMarker{q},'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='w');
                end
                hold on;

            saveas(fig4,selected_cells_name);
            end
    close(fig21);
end

cd(savepath)
save('rect.mat','rect')   
cd(HyS_path)

%%

%% n loop decides which cell is being looked at
for n = 1:cellnum 
    %% K loop decides which timepoint is being looked at
    for K = ini:nfiles 
        filename = filenames{K};

        try
            %% 24/04/2024 
            %% Debug: this code crashed if region was resized to rectangle not square, therefore code
            %% has been "fixed" to set Rect (CellSize) to a square but this may be slightly different
            %% to the actual region being measured - needs to be optimised
            %% use point_coord_x(n)+lolimit
            CellX(n) = [point_coord_x(n) + lolimit];
            CellY(n) = [point_coord_y(n) + lolimit];
            CellSize(:,n) = rect{n};
            CellIsAt{:,n} = [CellX(n) CellY(n) CellSize(3,n) CellSize(4,n)];
            runningCellOut(:,:,K) = imresize((imcrop(runningCell{:,:,K},[CellIsAt{:,n}])),[130,130]);
%% Figure 7 shows the cell selected region running through timepoints as they are analysed
%% if the cell wanders out of view or other cells come into view, this image shows that and the cell data can be removed
            fig7 = figure(7);
            imshow(runningCellOut(:,:,K));
            F7(K) = getframe(fig7);   
            
            cell_video_name = ['selected_cell_' num2str(n) ' '];
            v4 = VideoWriter(cell_video_name, 'MPEG-4'); %% I think you can change the format e.g. mp4
            v4.FrameRate = framerate;

            open(v4)
                writeVideo(v4,F7)
            close(v4);
         
        catch ME
            fprintf('Warning: problem importing file "%s"\n', filename);
        end
        K
    
        cellFolder = ['cell ' num2str(n)];
        cellSave = [savepath cellFolder];


%% check if folder exists for each individual cell (it should not exist unless two runs are started simultaneously) and create folder if it doesn't exist
        if not(isfolder([savepath cellFolder]))
            mkdir([savepath cellFolder])
        end

        cd(cellSave);
    
%% calculate average resonance wavelength of each ROI
%% set the map to the region containing the cell
        map1 = imcrop(imcrop(hypstore(:,:,K),location{n}),rect{n});  % new code to use cropped region
        map1_pre_correction = map1;

%% search hypstore and remove any pixels with intensity higher than the variable set at the start hyp_cell_cutoff
        imagemax{K} = max(map1);
        maxImageIntensity{K} = max(imagemax{K});
        Background{K} = wav_ini + step;

%% use average of cell ROI to decide what to replace
        ROImean = median(median(map1));
        CellIndicator{K} = (ROImean + 2.5);

%% replace any pixels that are resonating higher than the set cell threshold
        ROImeanreplace = mean(map1(map1 < CellIndicator{K})); %% this puts the mean of the remaining pixels into the cell pixels 
       
        if maxImageIntensity{K} > CellIndicator{K}    
            map1(map1 > CellIndicator{K}) = ROImeanreplace;
        end

%% lower limit can be used to make sure that things resonating under the lowest wavelength measured are not incorrectly labelled as lowest wavelength
%% This line MUST be removed if the wavelength span is reduced from that imaged because all signal lower than the new minimum will be set to the mean, skewing the result upwards
        % map1(map1 <= Background{K}) = ROImeanreplace;

%% if needed, this outputs raw data without the cell contact points replaced
    % hyp_raw_table_name = ['hyp_raw_cell_' num2str(n) '_time_' num2str(K) '.csv'];
    % hyp_raw_table = table(map1_pre_correction);
    % writetable(hyp_raw_table, hyp_raw_table_name);

%% insert an output for the mean and stdev for each image once all are collected (i.e. K is at nfiles)
        hyp_mean(K) = mean(mean(map1));
        hyp_stdev(K) = std(std(map1));
        hyp_median(K) = median(median(map1));

%% calculate resonance shift zeroed to the first timepoint
        hyp_res_shift(K) = hyp_mean(K) - hyp_mean(1);
        hyp_res_shift_median(K) = hyp_median(K) - hyp_median(1);

        if K == nfiles
%% adding variable to use to output all cell means to a single file
            hyp_mean_store(:,n) = hyp_mean;
            
            hyp_mean_table_name = ['hyp_mean_cell_' num2str(n) '.csv'];
            hyp_mean_table = table(hyp_mean);
            writetable(hyp_mean_table, hyp_mean_table_name);

            hyp_stdev_table_name = ['hyp_stdev_cell_' num2str(n) '.csv'];
            hyp_stdev_table = table(hyp_stdev);
            writetable(hyp_stdev_table, hyp_stdev_table_name);

            hyp_res_shift_mean_table_name = ['hyp_res_shift_mean_zero_cell_' num2str(n) '.csv'];
            hyp_res_shift_mean_table = table(hyp_res_shift);
            writetable(hyp_res_shift_mean_table, hyp_res_shift_mean_table_name);

            if OutputAllData == 'y'
                hyp_median_table_name = ['hyp_median_cell_' num2str(n) '.csv'];
                hyp_median_table = table(hyp_median);
                writetable(hyp_median_table, hyp_median_table_name);

                hyp_res_shift_median_table_name = ['hyp_res_shift_median_zero_cell_' num2str(n) '.csv'];
                hyp_res_shift_median_table = table(hyp_res_shift_median);
                writetable(hyp_res_shift_median_table, hyp_res_shift_median_table_name);
            end
        end
%% end of mean and stdev output

%% figure 8 shows two hyperspectral images, the cell region before any correction is done and the cell region after
%% the removal of any signal due to cell contact (using variable CellIndicator that uses ROImean (actually median value) plus a shift 
%% representative of cell contact e.g. 1.5nm - the size of this needs to be reliably determined
        fig8 = figure(8);
        fig8.Position(1:4) = [100 200 1000 600];
    
        subplot(1,2,1) 
            imshow(map1_pre_correction)
            shading interp
            caxis([colaxis(1) colaxis(2)]);
            colormap(jet)
            colorbar
            title('Cell ROI pre-correction','FontSize',12);

        subplot(1,2,2) 
              imshow(map1)
            shading interp
            caxis([colaxis(1) colaxis(2)]);
            colormap(jet)
            colorbar
            text_3 = [num2str(t_interval(K),'%0.1f') ' h   '];
            title(text_3,'FontSize',12);
            

        F8(K) = getframe(fig8);   
            cell_video_name = ['video_cell_' num2str(n) ' '];
            v3 = VideoWriter(cell_video_name, 'MPEG-4'); %% I think you can change the format e.g. mp4
            v3.FrameRate = framerate;
        
        open(v3)
            writeVideo(v3,F8)
        close(v3);
    

%% This normalises the mean at a timepoint against the mean at the first timepoint (n.b. will go negative if there is drift)
        av(K) = mean(mean(map1));
        av_norm(K) = av(K)-av(1);
        av_norm_stdev(K) = std(av(K)-av(1));
        av_median(K) = median(median(map1));
        av_median_norm(K) = av_median(K) - av_median(1);
        av_mean_zeroed_ini_comp_norm(K) = (((av(K) - wav_ini) - (av(1) - wav_ini)) + wav_ini);
        av_mean_zeroed_ini_comp_norm_stdev(K) = std(((av(K) - wav_ini) - (av(1) - wav_ini)) + wav_ini);


%% Output all pixels for each map region examined, only output ref on last loop

        if OutputAllData == 'y'
            hyp_av_table_name = ['hyp_av_cell_' num2str(n) '_time_' num2str(K) '.csv'];
            hyp_av_table = table(map1);
            writetable(hyp_av_table, hyp_av_table_name);
        end

        av_normalised(K) = av(K) - av(1);
        av_median_normalised(K) = av_median(K) - av_median(1);

%% set variable to use to calculate summary data
        for Phen = 1:K
            av_means(Phen,n) = av(Phen);
        end

%% output raw and processed data files
        if K == nfiles
            normalised_cell(:,n) = av_normalised;
            median_normalised_cell(:,n) = av_median_normalised;

            if OutputAllData == 'y'
                av_mean_norm_table_name = ['av_mean_zeroed_norm_cell_' num2str(n) '.csv'];
                av_mean_norm_table = table(av_norm);
                writetable(av_mean_norm_table, av_mean_norm_table_name);

                av_mean_norm_ini_comp_table_name = ['av_mean_zeroed_ini_comp_norm_cell_' num2str(n) '.csv'];
                av_mean_norm_ini_comp_table = table(av_mean_zeroed_ini_comp_norm);
                writetable(av_mean_norm_ini_comp_table, av_mean_norm_ini_comp_table_name);
           
                av_mean_norm_stdev_table_name = ['av_mean_norm_stdev_cell_' num2str(n) '.csv'];
                av_mean_norm_stdev_table = table(av_norm_stdev);
                writetable(av_mean_norm_stdev_table, av_mean_norm_stdev_table_name);

                av_mean_norm_ini_comp_stdev_table_name = ['av_mean_norm_ini_comp_stdev_cell_' num2str(n) '.csv'];
                av_mean_norm_ini_comp_stdev_table = table(av_mean_zeroed_ini_comp_norm_stdev);
                writetable(av_mean_norm_ini_comp_stdev_table, av_mean_norm_ini_comp_stdev_table_name);
            end    

 %% output file with summary data for mean resonances per timepoint split by cell and per cell by timepoint
            summaryFolder = ['data summary '];
            summarySave = [savepath summaryFolder];
            if not(isfolder([savepath summaryFolder]))
                mkdir([savepath summaryFolder])
            end
            cd(summarySave);
 
            summary_cell_means(:,n) = mean(av_means(:,n),1);
            av_cell_mean_table_name = ['SUMMARY_DATA_av_cell_mean.csv'];
            av_cell_mean_table = table(summary_cell_means);
            writetable(av_cell_mean_table, av_cell_mean_table_name);

            summary_cell_stdevs(:,n) = std(av_means(:,n),0,1);
            av_cell_stdevs_table_name = ['SUMMARY_DATA_av_cell_stdevs.csv'];
            av_cell_stdevs_table = table(summary_cell_stdevs);
            writetable(av_cell_stdevs_table, av_cell_stdevs_table_name);

            summary_timepoint_means = mean(av_means,2);
            av_timepoint_mean_table_name = ['SUMMARY_DATA_av_timepoint_mean.csv'];
            av_timepoint_mean_table = table(summary_timepoint_means);
            writetable(av_timepoint_mean_table, av_timepoint_mean_table_name);

            summary_timepoint_stdevs = std(av_means,0,2);
            av_timepoint_stdevs_table_name = ['SUMMARY_DATA_av_timepoint_stdevs.csv'];
            av_timepoint_stdevs_table = table(summary_timepoint_stdevs);
            writetable(av_timepoint_stdevs_table, av_timepoint_stdevs_table_name);

            if n == cellnum
                hyp_mean_store_table_name = ['SUMMARY_DATA_total_hyp_means_all_cells.csv'];
                hyp_mean_store_table = table(hyp_mean_store);
                writetable(hyp_mean_store_table, hyp_mean_store_table_name);
            end

        end

%% Lisa's "lines" code, not sure currently how to incorporate this
        % map1_wiener = wiener2(wiener2(wiener2(map1,[20 20])));
        % map1_norm = (map1_wiener-av(K));
        % lines(:,K) = map1_norm(:,round(length(map1)/2));
   
%% end of K loop is here so following code after this occurs at max K for each cell
    end
    
    cd(savepath)
    save(strcat('av_',num2str(n)),'av_norm')

    av_norm = (smooth(av_norm));
    hist(n) = (av_norm(K));

%% Figure 9 shows Isabel's smoothed line plot for means
    fig9 = figure(9)
        plot(t_interval(1:fin_hist),av_norm)
        legend
        hold on
        xlabel('Time after activation [h]')
        ylabel('Mean resonance shift over time [nm]')
        title('Cell mean shift smoothed over time normalised to time 0','FontSize',12)
        set(gca,'Fontsize',15)
        if n == cellnum
            saveas(gca,[ 'cell av norm against time 1 ' num2str(n) ' ' hyp_over_time_name]);
        end

%% Figure 10 - relevant figure - shows actual raw means without smoothing but normalised to timepoint zero (i.e. values approx 0 to 3nm)
    fig10 = figure(10)
        plot(t_interval(1:fin_hist),av_normalised)
        legend
        hold on
        xlabel('Time after activation [h]')
        ylabel('Mean resonance shift above reference [nm]')
        title('Cell mean shift normalised to TIME 0','FontSize',12)
        set(gca,'Fontsize',15)
        if n == cellnum
            saveas(gca,['cell av norm against time zero ' num2str(n) ' ' hyp_mean_name]);
        end

%% Figure 11 - relevant figure - shows each line with no normalisation (i.e. actual wavelengths for peaks)
    fig11 = figure(11)
        plot(t_interval(1:fin_hist),av)
        legend
        hold on
        xlabel('Time after activation [h]')
        ylabel('Resonance shift Mean [nm]')
        title('Cell mean peak resonance','FontSize',12)
        set(gca,'Fontsize',15)
        if n == cellnum
            saveas(gca,['cell av ' num2str(n) ' ' hyp_mean_no_norm_name]);
        end

%% Figure 12 uses medians instead of means in case that is interesting - removed for now
    % fig12 = figure(12)
        % plot(t_interval(1:fin_hist),av_median)
        % legend
        % hold on
        % xlabel('Time after activation [h]')
        % ylabel('Resonance shift Median [nm]')
        % title('Cell median peak shift over time','FontSize',12)
        % set(gca,'Fontsize',15)
        % if n == cellnum
            % saveas(gca,['cell av median ' num2str(n) ' ' hyp_median_name]);
        % end
    
%% empty all the variables set in the K loop to save memory
    clear map1;
    clear refmap;
    clear refmap_pre_correction;
    clear map1_pre_correction;
    clear hyp_mean_table;
    clear hyp_stdev_table;
    clear hyp_median_table;
    clear hyp_mean;
    clear hyp_stdev;
    clear hyp_median;
    clear hyp_res_shift;
    clear av_median;

    %% this is the end of the n loop processing the cells one by one
end


cd(savepath)
 saveas(gca,res_over_t_name)  

%% Figure 14 outputs the cells as a histogram of shift in resonance peak to see how much cell heterogeneity there is, but not much use as the shift size delimiting is uncontrolled
fig14 = figure(14) 
histogram(hist,'BinWidth', 0.2) %
legend(legend_name)
xlabel('Resonance shift [nm]')
ylabel('#cells')
title('Cell mean minus time 0 ROI','FontSize',12)
set(gca,'Fontsize',15)
axis padded
saveas(gca,histogram_name)

cd(savepath)
save(histogram_workspace,'hist')  
%%


%% Figure 15 - relevant figure - outputs a boxplot of the means for each cell
fig15 = figure(15)
boxplot(normalised_cell)
title('Mean resonance shift per cell')
saveas(gca, boxplot_cell_means)
ylabel('Mean Resonance shift [nm]')
set(gca,'Fontsize',15);
%%

%% Figure 16 outputs a boxplot for the all the cell means averaged to show the "per condition" mean - not much use alone
fig16 = figure(16)
boxplot(normalised_cell(normalised_cell>0))
title('Cell averaged mean resonance shift')
saveas(gca, boxplot_total_mean)
ylabel('Mean Resonance shift [nm]')
set(gca,'Fontsize',15);
%%

%% figure 17 outputs a bar chart of the number of cells achieving a specific resonance shift e.g. 2 cells at 0.2nm, 4 cells at 0.4nm etc 
fig17 = figure(17)
errorbar(mean(normalised_cell,2), std(normalised_cell,0,2))
title('Mean overall resonance shift')
saveas(gca, plot_total_mean)
ylabel('Mean Resonance shift [nm]')
set(gca,'Fontsize',15);
%

%% Figure 18 and figure 19 removed as not helpful, but can be replaced as part of collecting all raw data if needed
% if OutputAllData == 'y'
%% output a boxplot of the medians for each cell
    % fig18 = figure(18)
    % boxplot(median_normalised_cell)
    % title('Median resonance shift per cell')
    % saveas(gca, boxplot_cell_medians)
    % ylabel('Median Resonance shift [nm]')
    % set(gca,'Fontsize',15);
%%

%% output a boxplot for all the medians averaged to show the "per condition" median
    % fig19 = figure(19)
    % boxplot(median_normalised_cell(median_normalised_cell>0))
    % title('Averaged median resonance shift')
    % saveas(gca, boxplot_total_median)
    % ylabel('Median Resonance shift [nm]')
    % set(gca,'Fontsize',15);
% end

%% the "lines" figure follows if that code is included earlier

% fig14 = figure(14)
% imshow(lines_all{1})
% fig14.Position(1:4) = [100 100 1600 500];
% pcolor(lines_all{1})
% shading interp
% caxis([-0.5 0.5]);
% colormap(jet);
% c = colorbar;
% c.Label.String = 'Resonance wavelength shift \Delta\lambda [nm]';
% title('Spatiotemporal \Delta\lambda Cube','FontSize',12)
% set(gca,'xticklabel','','yticklabel','')
% set(gca,'Fontsize',15)

%% do not close all if you want to look at the graphs as they will be shut when processing ends if you close all
% close all

end
