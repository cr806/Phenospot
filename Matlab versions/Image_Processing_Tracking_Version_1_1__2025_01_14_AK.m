%% Image processing script

%% Clear all variables and figures pre-run
clear all
close all

%% INPUT: if you want to include a loop to pre-read the next location, set the number of locations here or set to 1 for a single location
number_of_wells = 1;

for analyse_all_wells = 1:number_of_wells

%% Initialisations
% Copy directory of location where the wavelength sweeps are saved in
% folder 0 1 ... don't forget backslash

if analyse_all_wells == 1
    clear all
    close all
    HyS_path = 'D:\Alasdair\Phenospot data\2024-11-28\Main Experiment\Location_1\HyperSpectral\'; % this is where the hyperspectral data is saved in the folders 1,2,3,...
    root = HyS_path;
    PhC_path = 'D:\Alasdair\Phenospot data\2024-11-28\Main Experiment\Location_1\PhaseContrast\'; %This path has to be where the phase contrast images were saved
    savepath = 'D:\Alasdair\Phenospot data\2024-11-28\Main Experiment\Location_1\Output\'; %This can be any path where you want to save the final histogram workspaces for further processing, not the 2 above
    scriptpath = 'P:\Phenospot Data\Sensor Tests\Drift testing\scripts\';
    cellCoordinatesPath = 'D:\Alasdair\Phenospot data\2024-11-28\Main Experiment\Location_1\CellCoordinates\';
    cellCoordinatesFile = '2024-11-28 Location 1 Cell Coordinates 2.xlsx';
    analyse_all_wells = 1;
    number_of_wells = 1;
    %% add a flag to decide whether to crop the cell area, value n or y (lowercase) - not cropping doesn't work yet
    chooseToCrop = 'n';
    % chooseToCrop = 'y';
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
wav_ini = 623.0;
wav_fin = 633.2;

%% Enable following code for multiple locations that use different wavelengths but add an if loop based on analyse_all_wells number
%% Location 2
% wav_ini = 667;
% wav_fin = 675.2;

%% INPUT: set the steps between each wavelength in the images taken
% step = 0.5;
step = 0.2;
% step = 1.0;

%% set size of auto selected ROI around cell point, defined as a square with boxSize as pixels per side, and code will set all related variables
boxSize = 64;

%% INPUT: Define size of area to be selected around the cell
%% -64 and 128 is a square 128 x 128 with cell in centre
%% Used for 10X imaging
if boxSize == 128
    lolimit = -64;
    hilimit = 128;
end
%% Large region for non-cell specific selection
if boxSize == 600
    lolimit = -300;
    hilimit = 600;
end
%% -32 and 64 is a square 64 x 64 with cell in centre
%% Used for 4X imaging
if boxSize == 64
    lolimit = -32;
    hilimit = 64;
end
%% Used for 4X imaging
if boxSize == 40
    lolimit = -20;
    hilimit = 40;
end

%% 19/11/2024 set up the cell region size for setting rect - this supercedes the hi and lo limit above for analysis but they are still used for images and videos
cellThreshold = 1.5;


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
%% This is over-ridden by a count of the number of cells in the tracking file - included to catch any errors
% cellnum = 40; 
% cellnum = 8;
cellnum = 5; % edit to 5 for speed testing
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
    hyp_data_from_table = 'hyperspec data norm from table cut';
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
%% Set tally to enable matlab to tell you which timepoint you are importing for on screen notification
Tally = 0;

filename = filenames{1};
all_data_cell = cell(nfiles, 1);
basenames = cell(nfiles, 1);
all_data_cell{1} = imread(filename);
       
A = (all_data_cell{1});

%% read coordinates file formatted as TrackId (numeric), Frame (timepoint), Xcoord (centre) and Ycoord (centre)
cd(cellCoordinatesPath);
    coordinatesFile = strcat(cellCoordinatesPath,cellCoordinatesFile);
        temp = xlsread(coordinatesFile);
        cellCoordinates(:, :) = temp(:, :);
        clear temp        

%% number of cells signified by a count of unique cell identifiers in the input coordinates file
        [coordCounts, TrackIDs] = groupcounts(cellCoordinates(:,1));
        numberOfCells = numel(coordCounts);

%% initialise the cell counts and timepoint numbers to a max cells of 999 and the correct number of timepoints
cellMax = 999;
timeMax = fin;

%% Read in the coordinates file as a table and name the columns
startTable = readtable(coordinatesFile);
startTable.Properties.VariableNames = {'cellId', 'frame', 'xin', 'yin'};

%% Extract the unique numeric cell IDs
uniqueCellIds = unique(startTable.cellId);
numCells = numel(uniqueCellIds);


%% Initialise the new table with zeroes ready to be filled
newCellIds = repelem(uniqueCellIds, timeMax); % Repeat each cell ID for all frames
newFrames = repmat((1:timeMax)', numCells, 1); % Repeat frames for all cells
newTable = table(newCellIds, newFrames, zeros(numCells * timeMax, 1), zeros(numCells * timeMax, 1), ...
                 'VariableNames', {'cellId', 'frame', 'xin', 'yin'});

%% Populate xin and yin coordinate values
%% Add in lines for all the timepoints that are not recorded by cell tracker, using the most recent, or subsequent next, cell location
%% this ensures there is an entry for each cell at each timepoint, even though cellTracker only shows times that cells are found
for j = 1:numCells
    cellId = uniqueCellIds(j);
    previousXin = 0;
    previousYin = 0;

%% Find the first occurrence of xin and yin for the cellId 
    firstRowIdx = find(startTable.cellId == cellId, 1);
    if ~isempty(firstRowIdx)
        firstXin = startTable.xin(firstRowIdx);
        firstYin = startTable.yin(firstRowIdx);
    else
        firstXin = 0;
        firstYin = 0;
    end

    for frame = 1:timeMax
        %% Find the index for the current cellId and frame
        newRowIdx = find(newTable.cellId == cellId & newTable.frame == frame);

        %% Find the corresponding row in startTable (if it exists)
        startRowIdx = find(startTable.cellId == cellId & startTable.frame == frame, 1);

        if ~isempty(startRowIdx)
            %% If a row exists, update the xin and yin values
            previousXin = startTable.xin(startRowIdx);
            previousYin = startTable.yin(startRowIdx);
        elseif frame == 1
            %% If there is no entry for timepoint 1, set it to the first occurrence
            previousXin = firstXin;
            previousYin = firstYin;
        end

        %% Fill in the xin and yin values (use the most recent values) into the new Table
        newTable.xin(newRowIdx) = previousXin;
        newTable.yin(newRowIdx) = previousYin;
    end
end

%% Sort newTable by cellId (numerical order) and frame (ascending order) to get everything in the correct order again
newTable = sortrows(newTable, {'cellId', 'frame'});

%% ***** Debug code to display the result
% disp(newTable);
% disp(startTable);
%% ***** end debug code

for c = 1:length(uniqueCellIds)
    % Get the current cell ID
    currentCellId = uniqueCellIds(c);
    % Filter rows for the current cell ID
    currentCellData = newTable(newTable.cellId == currentCellId, :);
    % Extract xin and yin values for this cell
    xinValues(c,:) = currentCellData.xin;
    yinValues(c,:) = currentCellData.yin;
end



%% make sure the cell number is at least 1 by setting to the value coded at the start if it is less
    if numberOfCells < 1
        numberOfCells = cellnum
    else
        cellnum = numberOfCells
    end

%% Determine the number of times to loop for the whole new table - the maximum iterations for all cells
    cellIterationTotal = numel(newTable(:,1));

%% Extract the x and y coordinates from newtable along with the cellId and frame number, populating the variable using newTable
%%why am I not using xinvalues and yinvalues array for co-ordinates, then I can use cell number and frame number instead of the cellIterationTotal
%%remove this if I implement the other loop
    for cellLoc = 1 : cellIterationTotal
        coordinateX(cellLoc,1) = table2array(newTable(cellLoc,3));
        coordinateY(cellLoc,1) = table2array(newTable(cellLoc,4));
        cellLabel(cellLoc,1) = table2array(newTable(cellLoc,1));
        cellFrame(cellLoc,1) = newTable(cellLoc,2);
    end      

%% check there are x and y coordinate for everything
    try
        if numel(xinValues) ~= numel(yinValues)
            fprintf('Warning: cell coordinates do not match for x and y');
        end
    catch ME
        fprintf('Warning: problem retrieving cell coordinates')
    end

cd(root)
%% start main processing
%% initialise mapstore - other code is Isabel's
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
%% loop d is through the number of timepoints
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

%% Change directory to where images are saved
%% read in each image as an individual variable: im1,im2,etc.
    folstring = strcat(HyS_path,num2str(timemultiple),'/');
    cd(folstring)
    format = 'Image %d of %d imported \n';
    A = dir('*.TIFF');

    C = {A.name};
    cd(scriptpath)
    Aname = natsort(C);
    cd(folstring)

%% corrected order of xsize and ysize as they appeared to be inverted, although code worked anyway
    temp = importdata(string(Aname(1)));
    ysize = size(temp, 1);
    xsize = size(temp, 2);
    imstack = zeros(ysize, xsize, length(A));
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
%% altered to catch if there is no signal i.e. pos_initial is less than wav_ini
    if pos_initial < wav_ini
        caxis_left = pos_initial - wav_ini;
        caxis_right = wav_fin - pos_initial;
    else
        caxis_left = pos_initial - wav_ini - 2;
        caxis_right = wav_fin - pos_initial - 2;
    end

 
%% Display Figure 1 the resonance map for on-screen reference
    fig1 = figure(1);
    pcolor(wiener2(map,[2 2]))
    shading interp
    caxis([colaxis(1) colaxis(2)]);
    colormap(jet);
    colorbar
    title('Resonance map','FontSize',12)
    axis equal
    set(gca,'xticklabel','','yticklabel','')
    format = 'Plotting finished \n';
    fprintf(format);

%% set region to whole image
%% image co-ordinates should be 0,0 to 1460,1920 but coded to detect from image just in case it isn't
    region = imstack(1:ysize,1:xsize,1:end);

%% clear imstack to save memory
    clear imstack;

%% 2024-10-21 set map to whole image
    mapregion = map(1:ysize,1:xsize);

%% store all regions to assess later
    hypstore(:,:,d) = mapregion;
    reswav_std(d) = std(std(mapregion));

%% 2024-10-21 set ar to whole image as well -- not sure what ar is for, doesn't appear to be used
    ar = [0 ysize; xsize xsize];
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
%% sets the axes on the graph
    if reswav(1) > wav_ini
        caxis_left = reswav(1) - wav_ini - 2;
        caxis_right = wav_fin - reswav(1) - 2;
    else
        caxis_left = reswav(1) - wav_ini;
        caxis_right = 2;
    end

%% Figure 2 shows the selected ROI on the hyperspectral and phase contrast for each timepoint in turn, data output as video
    fig2 = figure(2);
    fig2.Position(1:4) = [100 200 1600 500];
    subplot(1,2,1)
        customJet = jet(256); % Create the colormap
        customJet = customJet .^ 0.8; % Adjust gamma for brightness
        mapflip = flipud(mapregion);
        pcolor(mapflip)
        shading interp
        caxis([colaxis(1) colaxis(2)]);
        colormap(gca, customJet);
        c = colorbar;
        c.Label.String = 'Resonance wavelength \lambda [nm]';
        title('Resonance map','FontSize',12)
        set(gca,'xticklabel','','yticklabel','')
        set(gca,'Fontsize',15)
        axis tight equal

    subplot(1,2,2)
        A = (all_data_cell{d});
        A = A(1:ysize,1:xsize);
        A = imadjust(medfilt2(A(:,:,1)));
%% store the phase contrast images sequentially in "running cell" so that the selected cells can be displayed later
%% this enables tracking of cells within the selected region to make sure they are present throughout the experiment
        runningCell{1,d} = flipud(A);
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
%% end of d loop for all timepoints, having retrieved and got max for all hyperspectrals
end
%%

cd(savepath)

%% 2024-11-14 outputs the whole map data from hypstore to a table for each timepoint then creates an array of those tables that can be used to get cell region data
for d = ini:fin
    hyp_raw_map_table_name = ['hyp_raw_map_' num2str(d) '.csv'];
    hyp_raw_map_table = table(hypstore(:,:,d));
    writetable(hyp_raw_map_table, hyp_raw_map_table_name);
    raw_Map_Array(:,:,d) = table2array(hyp_raw_map_table);
end

   
%% Preallocate a 3D array based on size of first element
[rows, cols] = size(raw_Map_Array(:,:,1)); % Size of the first 2D array

%% Fill the 3D array with data from each 2D array to consolidate
for combiner = 1:fin
    combined_Raw_Map_Array(:, :, combiner) = hypstore(:,:,combiner);
end

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

%% n loop for each cell selection, goes to line 810 (or thereabouts after edits)
for n = 1: numCells
    cellId = uniqueCellIds(n);
    cellId = cellId + 1;

    xLocs1(n,:) = xinValues(n,:);
    yLocs1(n,:) = yinValues(n,:);

    for frameNum = 1:fin
%% this is where ROI is selected on phase contrast
        format = 'Running my added code next \n';
        fprintf(format);
%% select cells area from the cropped image selected at the start
        flippedA = flipud(A);
%% take the coordinates saved earlier into the original code xcoord and ycoord
        xcoord(n,1) = xinValues(n,frameNum)
        ycoord(n,1) = yinValues(n,frameNum)

%% use lolimit and hilimit variables set at the start of the code to define area size 
        xlo = xcoord(n,1) + lolimit;
        xhi = hilimit;
        ylo = ycoord(n,1) + lolimit;
        yhi = hilimit;
%% point coord is the centre of the area, the actual location given for the cell
%% check Trackmate is giving the centre and not the edge
        point_coord_x(n) = xcoord(n,1);
        point_coord_y(n) = ycoord(n,1);
        cellMarker{n} = string(cellLabel(n,1) + 1);
        ROIsize =  [0,0,xhi,yhi];

%% set the array size based on the dimensions of the raw map array, used to create the box around the cells
        arrayWidth = xsize; % Replace with actual dimensions of combined_Raw_Map_array
        arrayHeight = ysize; % Replace with actual dimensions of combined_Raw_Map_array
        numFrames = fin; % Number of frames in the timelapse

%% Box size parameters - resetting box size based on lolimit but it is set at the start so this should match!
%% lolimit is negative offset from centrepoint, so abs makes positive and times 2 to get size to upper limit
        boxSize = abs(lolimit) * 2; % Box size (64x64 for lolimit = -32)
        centreX = round(xinValues(n,frameNum));
        centreY = round(yinValues(n,frameNum));
%% Skip if no valid coordinates
        if centreX == 0 && centreY == 0
            continue;
        end
%% Define the box boundaries 
        xStart = centreX + lolimit;
        xEnd = min(arrayWidth, centreX + lolimit + boxSize);
        yStart = centreY + lolimit;
        yEnd = min(arrayHeight, centreY + lolimit + boxSize);

%% reset negative values or zeroes to pixel 1 so it is on screen
        if xStart < 1
            xStart = 1
        end

        if xEnd < 1
            xEnd = 1
        end
        
        if yStart < 1
            yStart = 1
        end
        
        if yEnd < 1
            yEnd = 1
        end
        
        if n == 1
            loopNote = 0;
        end
        loopNote = loopNote + 1;
        fprintf('Cell %d:, Loop = %d, centreX = %.3f, centreY = %.3f, xStart = %.3f, xEnd = %.3f, yStart = %.3f\n, yEnd = %.3f', n, loopNote, centreX, centreY, xStart, xEnd, yStart, yEnd);

        %% Extract region for the current frame
        currentRegion = combined_Raw_Map_Array(yStart:yEnd, xStart:xEnd, frameNum);
%% debug 
        % current_region_table_name = ['current region_' num2str(n) '_time_' num2str(frameNum) '.csv'];
        % current_region_table = table(currentRegion);
        % writetable(current_region_table, current_region_table_name);

%% Extract region for frame minus one
        if frameNum > 1
            currentRegionMinusOne = hypstore(yStart:yEnd, xStart:xEnd, (frameNum - 1));
        else
%% Use the current frame region if frame is 1 because there is no minus 1
            currentRegionMinusOne = currentRegion;
        end

%% Cell signal correction to remove cell from currentRegion before calculating regionShift          
%% replace all pixel values over the threshold set at start with the mean of the other pixel values for currentRegion
%% Iterative process to calculate maxValue
%% use 1e-6 for perfect result, changed to 0.21 due to concern over specificity lower than wavelength interval
        tolerance = 1e-6; % Convergence tolerance
%% initialise prevMax to infinity
        prevMaxValue = inf;
%% initialise currentArray with the calculated shift
        currentRegionArray = currentRegion;

        while true
%% Compute mean of current valid values
            validValues = currentRegionArray(currentRegionArray <= prevMaxValue); % Exclude excess values
            meanValue = mean(validValues(:)); % Compute mean of valid values
%% Update maxValue with the mean plus the cell signal threshold
            maxValue = meanValue + cellThreshold;
%% Check for convergence and stop if the new maxValue is less than the tolerance set
            if abs(maxValue - prevMaxValue) < tolerance
                break;
            end
%% Update prevMaxValue so that the next comparison uses this, and at the end the maxValue will be the best one
            prevMaxValue = maxValue;
        end

% Replace values > maxValue with the final meanValue
        currentRegionArray(currentRegionArray > maxValue) = meanValue;
%% debug code
        % disp(regionShift);

%% Cell signal correction to remove cell from previous Region before calculating regionShift          
%% replace all pixel values over the threshold with the mean of the other pixel values
%% Iterative process to calculate maxValue
%% use 1e-6 for perfect result, changed to 0.21 due to concern over specificity lower than wavelength interval
        tolerance = 1e-6; % Convergence tolerance
%% initialise prevMax to infinity
        prevMaxValue = inf;
%% initialise currentArray with the calculated shift
        previousRegionArray = currentRegionMinusOne;

        while true
%% Compute mean of current valid values
            validValues = previousRegionArray(previousRegionArray <= prevMaxValue); % Exclude excess values
            meanValue = mean(validValues(:)); % Compute mean of valid values
%% Update maxValue with the mean plus the cell signal threshold
            maxValue = meanValue + cellThreshold;
%% Check for convergence and stop if the new maxValue is less than the tolerance set
            if abs(maxValue - prevMaxValue) < tolerance
                break;
            end
%% Update prevMaxValue so that the next comparison uses this, and at the end the maxValue will be the best one
            prevMaxValue = maxValue;
        end

%% Replace values > maxValue with the final meanValue
        previousRegionArray(previousRegionArray > maxValue) = meanValue;
%% debug code
        % disp(regionShift);

%% calculate region mean minus previous region mean
%% Compute the difference from previous regon value (region shift) then empty currentRegion to make it clear for reuse
        pre_correction_regionStore{cellId,frameNum} = currentRegionArray;
        pre_correction_Mean = mean(pre_correction_regionStore{cellId, frameNum}(:));
        pre_correction_cell_Means(cellId,frameNum) = pre_correction_Mean;

%% adjust regionshift to difference between timepoints at selected location
        regionShift = currentRegionArray - previousRegionArray;

        post_correction_regionStore{cellId,frameNum} = regionShift;
        post_correction_Mean = mean(post_correction_regionStore{cellId, frameNum}(:));
        post_correction_cell_Means(cellId,frameNum) = post_correction_Mean;

%% debug code to see what has been filled in and display on-screen
        fprintf('CellId: %d, Frame: %d\n', cellId, frameNum);
        disp('Current Region mean:');
        disp(mean(currentRegion(:)));
        disp('Current Region Minus One mean:');
        disp(mean(currentRegionMinusOne(:)));

%% debug
%% write out corrected data table
        current_region_corrected_table_name = ['current_region_corrected_cell_' num2str(cellId) '_time_' num2str(frameNum) '.csv'];
        current_region_corrected_table = table(regionShift);
        writetable(current_region_corrected_table, current_region_corrected_table_name);

        regionStore{cellId,frameNum} = regionShift;

%% examine here to see that when the cell has moved, the region of cell means t-1 is the same location as the current region and not the previous location
%% Compute the mean for the current region 
        currentMean = mean(regionStore{cellId, frameNum}(:));
%% Store the mean of regionShift in the cellMeans array
        cellMeans(cellId,frameNum) = currentMean;
%% for the first timepoint, make cumulative means 0
        if frameNum == 1
            meanDifferences(cellId,frameNum) = 0;
            cumulativeMeans(cellId,frameNum) = cellMeans(cellId,frameNum) - cellMeans(cellId,frameNum);
        else
            frameMinusOne = frameNum - 1;
            meanDifferences(cellId,frameNum) = cellMeans(cellId,frameNum) - cellMeans(cellId,frameMinusOne);
            cumulativeMeans(cellId,frameNum) = cellMeans(cellId,frameNum) + cumulativeMeans(cellId,frameMinusOne);
        end
%% end of framenum loop
    end

%% Debugging outputs displayed on screen
    fprintf('Timepoint %d: Current Mean = %.3f, Difference = %.3f, Cumulative = %.3f\n', ...
            frameNum, cellMeans(cellId,frameNum), meanDifferences(cellId,frameNum), cumulativeMeans(cellId,frameNum));
    fprintf('Frame %d: x [%d:%d], y [%d:%d], currentRegion Mean = %.3f\n', ...
            frameNum, xStart, xEnd, yStart, yEnd, mean(currentRegion(:)));
%% meanDifferences contains the difference in means between consecutive timepoints - not used for anything

%% cellMeans now contains the mean of the region for each timepoint
    [HShighlights{n}, location{n}] = imcrop(mapregion,[xStart,yStart,xhi,yhi]); 

%% Figure 4 shows the phase contrast of just the cell region to be analysed based on the point click previously
    rect{n} = ROIsize;

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
                imshow(A)
                for q = 1:cellnum
                    h = drawpoint('Position',[point_coord_x(q) point_coord_y(q)],'Label',cellMarker{q},'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='w');
                end
                hold on;

            saveas(fig4,selected_cells_name);
    end
%% end the n loop from line 592 or thereabouts
end

%% Isabel's save - not biologically useful data
cd(savepath)
save('rect.mat','rect')   
cd(HyS_path)


%% Loop through each cell and frame
for cellIdx = 1:numCells
    xLocs(cellIdx,:) = xinValues(cellIdx,:);
    yLocs(cellIdx,:) = yinValues(cellIdx,:);

    for frameIdx = 1:numFrames
%% Get the coordinates for the crop center
        centreX = round(xLocs(cellIdx, frameIdx));
        centreY = round(yLocs(cellIdx, frameIdx));

%% Skip if no valid coordinates (e.g., NaN or zero)
        if isnan(centreX) || isnan(centreY) || centreX == 0 || centreY == 0
            continue;
        end



%% Define starting coordinates
        xStart2 = centreX + lolimit;
        yStart2 = centreY + lolimit;

%% Ensure the region stays within bounds
        if xStart2 < 1
            xStart2 = 1; % Adjust if it goes out of the left boundary
        end
        if yStart2 < 1
            yStart2 = 1; % Adjust if it goes out of the top boundary
        end

        if (xStart2 + boxSize - 1) > arrayWidth
            boxSize = arrayWidth - xStart2 + 1; % Adjust boxSize if it goes out of the right boundary
        end
        if (yStart2 + boxSize - 1) > arrayHeight
            boxSize = arrayHeight - yStart2 + 1; % Adjust boxSize if it goes out of the bottom boundary
        end

%% Extract the cropped region from the image
        cropLoc{cellIdx,frameIdx} = [xStart2, yStart2, boxSize, boxSize];
        try
            croppedRegion = all_data_cell{frameIdx};  % new code to use cropped region
            croppedRegion = imcrop(croppedRegion,cropLoc{cellIdx,frameIdx});  % new code to use cropped region
        catch ME
            fprintf('Warning: problem cropping image');
        end

        try
            fig7 = figure(7);
            imshow(croppedRegion);
            F7(frameIdx) = getframe(fig7);   
            cell_video_name = ['selected_cell_' num2str(cellIdx) ' '];
            v4 = VideoWriter(cell_video_name, 'MPEG-4'); %% I think you can change the format e.g. mp4
            v4.FrameRate = framerate;
            open(v4)
                writeVideo(v4,F7)
            close(v4);
        catch ME
            fprintf('Warning: problem displaying cropped image', filename);
        end
    
        cellFolder = ['cell ' num2str(cellIdx)];
        cellSave = [savepath cellFolder];
%% check if folder exists for each individual cell (it should not exist unless two runs are started simultaneously) and create folder if it doesn't exist
        if not(isfolder([savepath cellFolder]))
            mkdir([savepath cellFolder])
        end
        cd(cellSave);

%% set up a for loop for x to be xlo to xhi and y to be ylo to yhi and retrieve the data
%% calculate average resonance wavelength of each ROI
%% set the map to the region containing the cell
        
        cropLoc2{cellIdx,frameIdx} = [xStart2, yStart2, boxSize, boxSize];
        map1 = imcrop(hypstore(:,:,frameIdx),cropLoc2{cellIdx,frameIdx});  % new code to use cropped region
        map1_pre_correction = map1;

%% debug code
%        map1_table_name = ['map1' num2str(cellIdx) ' ' num2str(frameIdx) '.csv'];
%        map1_table = table(hypstore(:,:,frameIdx));
%        writetable(map1_table, map1_table_name);
%        map2_table_name = ['map2' num2str(cellIdx) ' ' num2str(frameIdx) '.csv'];
%        map2_table = table(map1);
%        writetable(map2_table, map2_table_name);
    

%% search map1 (cut down from hypstore) and remove any pixels with intensity higher than the variable set at the start hyp_cell_cutoff
        imagemax{frameIdx} = max(map1);
        maxImageIntensity{frameIdx} = max(imagemax{frameIdx});
        Background{frameIdx} = wav_ini + step;

%% replace all pixel values over the threshold with the mean of the other pixel values
%% Iterative process to calculate maxValue as performed earlier
        cellTolerance = 1e-6; % Convergence tolerance
        cellPrevMaxValue = inf; % Initialize with a large value
        cellCurrentArray = map1; % Start with the full dataset

        while true
%% Compute mean of current valid values
            cellValidValues = cellCurrentArray(cellCurrentArray <= cellPrevMaxValue); % Exclude excess values
            cellMeanValue = mean(cellValidValues(:)); % Compute mean of valid values
%% Update maxValue
            cellMaxValue = cellMeanValue + cellThreshold;
%% Check for convergence
            if abs(cellMaxValue - cellPrevMaxValue) < cellTolerance
                break; % Stop iteration if maxValue converges
            end
%% Update prevMaxValue and currentArray
            cellPrevMaxValue = cellMaxValue;
        end

%% Replace values > maxValue with the final meanValue so that cell signal is removed
        map1(map1 > cellMaxValue) = cellMeanValue;

%% if needed, this outputs raw data without the cell contact points replaced and with them replaced
        hyp_raw_table_name = ['hyp_raw_cell_' num2str(cellIdx) '_time_' num2str(frameIdx) '.csv'];
        hyp_raw_table = table(map1_pre_correction);
        writetable(hyp_raw_table, hyp_raw_table_name);
        hyp_corrected_table_name = ['hyp_corrected_cell_' num2str(cellIdx) '_time_' num2str(frameIdx) '.csv'];
        hyp_corrected_table = table(map1);
        writetable(hyp_corrected_table, hyp_corrected_table_name);

%% insert an output for the mean and stdev for each image once all are collected (i.e. K is at nfiles)
        hyp_mean(frameIdx) = mean(map1(:));
        hyp_stdev(frameIdx) = std(std(map1));
        hyp_median(frameIdx) = median(median(map1));

%% calculate resonance shift zeroed to the first timepoint - this can be converted to cumulative using the index K-1 instead of 1 (with other changes)
        hyp_res_shift(frameIdx) = hyp_mean(frameIdx) - hyp_mean(1);
        hyp_res_shift_median(frameIdx) = hyp_median(frameIdx) - hyp_median(1);

%% at the end, output the total data for the cell before moving to the next cell
        if frameIdx == nfiles
            hyp_mean_store(:,cellIdx) = hyp_mean;
            hyp_mean_table_name = ['hyp_mean_cell_' num2str(cellIdx) '.csv'];
            hyp_mean_table = table(hyp_mean);
            writetable(hyp_mean_table, hyp_mean_table_name);
%%
            hyp_stdev_table_name = ['hyp_stdev_cell_' num2str(cellIdx) '.csv'];
            hyp_stdev_table = table(hyp_stdev);
            writetable(hyp_stdev_table, hyp_stdev_table_name);
%%
            hyp_res_shift_mean_table_name = ['hyp_res_shift_mean_zero_cell_' num2str(cellIdx) '.csv'];
            hyp_res_shift_mean_table = table(hyp_res_shift);
            writetable(hyp_res_shift_mean_table, hyp_res_shift_mean_table_name);

%% if desired, output all of the tiny bits of data collected - probably not used much in reality        
            if OutputAllData == 'y'
                hyp_median_table_name = ['hyp_median_cell_' num2str(cellIdx) '.csv'];
                hyp_median_table = table(hyp_median);
                writetable(hyp_median_table, hyp_median_table_name);
                hyp_res_shift_median_table_name = ['hyp_res_shift_median_zero_cell_' num2str(cellIdx) '.csv'];
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
            subplot(2,3,1)
                imshow(mapregion, [])
                h = drawpoint('Position',[xStart2 yStart2],'Label',[num2str(cellIdx) ', ' num2str(frameIdx)],'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='k');
                hold on;
                shading interp
                caxis([colaxis(1) colaxis(2)]);
                colormap(gca,jet)
                c = colorbar;
                c.Label.String = 'Resonance wavelength \lambda [nm]';
               
            subplot(2,3,2) 
                imshow(map1_pre_correction, [])
                shading interp
                caxis([colaxis(1) colaxis(2)]);
                colormap(gca,jet)
                colorbar
                title('Cell ROI pre-correction','FontSize',12);

            subplot(2,3,3) 
                imshow(map1, [])
                shading interp
                caxis([colaxis(1) colaxis(2)]);
                colormap(gca,jet)
                colorbar
                text_3 = [num2str(t_interval(K),'%0.1f') ' h   '];
                title(text_3,'FontSize',12);
            
            
            subplot(2,3,4)
               try
                    croppedShow = imadjust(medfilt2(all_data_cell{frameIdx}));  % new code to use cropped region
                    imshow(croppedShow, [])
                    h = drawpoint('Position',[xStart2 yStart2],'Label',[num2str(cellIdx) ', ' num2str(frameIdx)],'LabelAlpha',0,MarkerSize=5,Color='m',LabelTextColor='w');
                    hold on;
               catch ME
                    fprintf('Warning: problem showing cropped region', filename);
               end
% 
            subplot(2,3,5)
                imshow(croppedRegion, [])
                colormap(gca,gray)
                title('Cell ROI','FontSize',12);
            

            subplot(2,3,6)
                if frameIdx == 1 % Assumes frameIdx resets to 1 for each new cell
                    cla; % Clear only the current subplot axes to show just next cell
                end
                colorMap = lines(numCells);
                currentColor = colorMap(cellIdx, :);
                hline = plot(t_interval(1:frameIdx), cumulativeMeans(cellIdx, 1:frameIdx), 'LineWidth', 1.5, 'Color', currentColor);
                legend(hline, ['Cell ' num2str(cellIdx)], 'Location', 'best');
                hold on
                xlabel('Time [h]')
                ylabel('Resonance shift [nm]')
                title('Cumulative res shift','FontSize',8)
                set(gca,'Fontsize',8)
    

            F8(frameIdx) = getframe(fig8);   
                cell_video_name = ['video_cell_' num2str(cellIdx) ' '];
                v3 = VideoWriter(cell_video_name, 'MPEG-4'); %% I think you can change the format e.g. mp4
                v3.FrameRate = framerate;
        
            open(v3)
                writeVideo(v3,F8)
            close(v3);
    

%% This normalises the mean at a timepoint against the mean at the first timepoint (n.b. will go negative if there is drift)
            av(frameIdx) = mean(map1(:));
            av_norm(frameIdx) = av(frameIdx)-av(1);
 
            if frameIdx == 1
                av_cumul(1) = 0
            end

            if frameIdx > 1
                av_cumul(frameIdx) = av_cumul(frameIdx-1) + (av(frameIdx)-av(frameIdx-1));
            end
            
            av_norm_stdev(frameIdx) = std(av(frameIdx)-av(1));
            av_median(frameIdx) = median(median(map1));
            av_median_norm(frameIdx) = av_median(frameIdx) - av_median(1);
            av_mean_zeroed_ini_comp_norm(frameIdx) = (((av(frameIdx) - wav_ini) - (av(1) - wav_ini)) + wav_ini);
            av_mean_zeroed_ini_comp_norm_stdev(frameIdx) = std(((av(frameIdx) - wav_ini) - (av(1) - wav_ini)) + wav_ini);

%% Output all pixels for each map region examined, only output ref on last loop
            if OutputAllData == 'y'
                hyp_av_table_name = ['hyp_av_cell_' num2str(cellIdx) '_time_' num2str(K) '.csv'];
                hyp_av_table = table(map1);
                writetable(hyp_av_table, hyp_av_table_name);
            end

            av_normalised(frameIdx) = av(frameIdx) - av(1);
            av_median_normalised(frameIdx) = av_median(frameIdx) - av_median(1);

%% set variable to use to calculate summary data
            for Phen = 1:frameIdx
                av_means(Phen,cellIdx) = av(Phen);
            end

%% output raw and processed data files
            if frameIdx == nfiles
                normalised_cell(:,cellIdx) = av_normalised;
                median_normalised_cell(:,cellIdx) = av_median_normalised;
                if OutputAllData == 'y'
                    av_mean_norm_table_name = ['av_mean_zeroed_norm_cell_' num2str(cellIdx) '.csv'];
                    av_mean_norm_table = table(av_norm);
                    writetable(av_mean_norm_table, av_mean_norm_table_name);
     
                    av_mean_norm_ini_comp_table_name = ['av_mean_zeroed_ini_comp_norm_cell_' num2str(cellIdx) '.csv'];
                    av_mean_norm_ini_comp_table = table(av_mean_zeroed_ini_comp_norm);
                    writetable(av_mean_norm_ini_comp_table, av_mean_norm_ini_comp_table_name);
               
                    av_mean_norm_stdev_table_name = ['av_mean_norm_stdev_cell_' num2str(cellIdx) '.csv'];
                    av_mean_norm_stdev_table = table(av_norm_stdev);
                    writetable(av_mean_norm_stdev_table, av_mean_norm_stdev_table_name);
    
                    av_mean_norm_ini_comp_stdev_table_name = ['av_mean_norm_ini_comp_stdev_cell_' num2str(cellIdx) '.csv'];
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
     
                summary_cell_means(:,cellIdx) = mean(av_means(:,cellIdx),1);
                av_cell_mean_table_name = ['SUMMARY_DATA_av_cell_mean.csv'];
                av_cell_mean_table = table(summary_cell_means);
                writetable(av_cell_mean_table, av_cell_mean_table_name);
    
                summary_cell_stdevs(:,cellIdx) = std(av_means(:,cellIdx),0,1);
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
    
                if cellIdx == cellnum
                    hyp_mean_store_table_name = ['SUMMARY_DATA_total_hyp_means_all_cells.csv'];
                    hyp_mean_store_table = table(hyp_mean_store);
                    writetable(hyp_mean_store_table, hyp_mean_store_table_name);
                end
                if cellIdx == cellnum
                    moving_means_store_table_name = ['SUMMARY_DATA_cumul_moving_means_all_cells.csv'];
                    moving_means_store_table = table(cumulativeMeans);
                    writetable(moving_means_store_table, moving_means_store_table_name);
                end
            end

%% Lisa's "lines" code, not sure currently how to incorporate this
        % map1_wiener = wiener2(wiener2(wiener2(map1,[20 20])));
        % map1_norm = (map1_wiener-av(K));
        % lines(:,K) = map1_norm(:,round(length(map1)/2));
   
%% end of FRAME IDx loop is here so following code after this occurs at max for each cell but within n loop
        end
    
        cd(savepath)
        save(strcat('av_',num2str(cellIdx)),'av_norm')
    
        % av_norm = (smooth(av_norm));
        hist(cellIdx) = (av_norm(frameIdx));
    
%% Figure 9 shows Isabel's smoothed line plot for means
        % fig9 = figure(9)
        %     plot(t_interval(1:fin_hist),av_norm)
        %     legend
        %     hold on
        %     xlabel('Time after activation [h]')
        %     ylabel('Mean resonance shift over time [nm]')
        %     title('Cell mean shift smoothed over time normalised to time 0','FontSize',12)
        %     set(gca,'Fontsize',15)
        %     if cellIdx == cellnum
        %         saveas(gca,[ 'cell av norm against time 1 ' num2str(cellIdx) ' ' hyp_over_time_name]);
        %     end
    
%% Figure 10 - relevant figure - shows actual raw means without smoothing but normalised to timepoint zero (i.e. values approx 0 to 3nm)
        fig10 = figure(10)
            currentColor = colorMap(cellIdx, :);
            plot(t_interval(1:fin_hist),av_normalised, 'LineWidth', 1.5, 'Color', currentColor)
            legend
            hold on
            xlabel('Time after activation [h]')
            ylabel('Mean resonance shift above reference [nm]')
            title('Cell mean shift normalised to TIME 0','FontSize',12)
            set(gca,'Fontsize',15)
            if cellIdx == cellnum
                saveas(gca,['cell av norm against time zero ' num2str(cellIdx) ' ' hyp_mean_name]);
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
            if cellIdx == cellnum
                saveas(gca,['cell av ' num2str(cellIdx) ' ' hyp_mean_no_norm_name]);
            end
    
%% Figure 12 show output from cumulative means that should match figure 10
        if cellIdx == cellnum
            fig12 = figure(12)
                plot(t_interval(1:fin_hist),cumulativeMeans)
                legend
                hold on
                xlabel('Time after activation [h]')
                ylabel('Resonance shift Mean [nm]')
                title('Cell mean peak resonance','FontSize',12)
                set(gca,'Fontsize',15)
                % if n == cellnum
                    saveas(gca,['cell av ' num2str(cellIdx) ' ' hyp_data_from_table]);
                % end
         end
%% debug code
% disp(mean(currentRegion(:))); % Raw region mean
% disp(mean(regionStore{cellId, frameCount}(:))); % Corrected region mean
% 
% disp(['CumulativeMeans: ', num2str(cumulativeMeans(cellId, frameCount))]);
% disp(['Av_Cumul: ', num2str(av_cumul(frameIdx))]);
% 
% disp('Map1 corrections:');
% disp(cropLoc2{cellIdx, frameIdx}); % Map1 uses cropLoc2 for ROI
% 
% disp('RegionShift corrections:');
% disp(regionStore{cellIdx, frameIdx}); % regionShift uses regionStore

% disp('Tracking ROI:');
% disp(debugCoordinates1);
% disp('Static ROI:');
% disp(debugCoordinates2);

% disp('RegionStore ROI:');
% disp([yStart, yEnd, xStart, xEnd]);
% 
% disp('ROI used for map1:');
% disp(cropLoc2{cellIdx, frameIdx});
% 
% disp('ROI used for regionShift:');
% disp(regionStore{cellIdx, frameIdx});

%% empty all the variables set in the K loop to save memory
      clear refmap;
      clear refmap_pre_correction;
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

%% end of the loop that started at the very beginning, used for processing multiple wells simultaneously
end
