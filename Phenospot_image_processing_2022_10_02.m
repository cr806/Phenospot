%% Image processing script, Phenospot for single location

clear all
close all

%% Initialisations
% Copy directory of location where the wavelength sweeps are saved in
% folder 0 1 ... don't forget backslash
%savepath should be where phase contrast images are saved 

HyS_path = 'D:\Isabel\Testdata_cube_macrophages\2022_07_20_all\Location_1\Hyperspectral\'; % this is where the hyperspectral data is saved in the folders 1,2,3,...
root = HyS_path;
PhC_path = 'D:\Isabel\Testdata_cube_macrophages\2022_07_20_all\Location_1\Phasecontrast\'; %This path has to be where the phase contrast images were saved
savepath = 'D:\Isabel\Testdata_cube_macrophages\2022_07_20_all\Location_1\'; %This can be any path where you want to save the final histogram workspaces for further processing, not the 2 above
scriptpath = 'D:\Isabel\Matlab_scripts\Macrophages\';

%% Define name of file where all hyperspectral maps are stored and saved

mapfilename = 'mapstore_test.mat';

%%
% define wavelength sweep range
wav_ini = 670;
wav_fin = 690;
step = 0.2;
%%

%How many hours after cell activation was the first image taken?
act_time = 0; %in h
% How many points do you want to consider for the average resonance shift
% over time?
fin = 5;

%%
    videoname = 'loc01_ROI_av.avi';
    v.FrameRate = 2; %frames per second
    
    %%
    cellnum = 20; % how many cells are you looking at in the FOV, this is fixed to 20 here
    %%
    histogram_name = 'loc_01_ROI_av_hist.png';
    res_over_t_name = 'res_curves_wav_t_loc_01_ROI_av.png';
    boxplot_name = 'boxplot_loc_01_ROI_av.png'
    
    histogram_workspace = 'loc_01_hist_ROI_av.mat'; 
    legend_name = 'well 01 ROI av';
    
    %%
    cd(PhC_path)
data = dir('*.tiff');
filenames = fullfile( {data.folder}, {data.name});
nfiles = length(filenames);
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

for i = 1:nfiles
info = imfinfo(filenames{i});
timestamp{i} = datevec(info.FileModDate);
t_interval(i) = (etime(timestamp{i},timestamp{1}))./60/60 + act_time; %in hours
end

ini = 1;
ending = nfiles;

%% Look at phase contrast images and save video --> visual check #1
    for K = ini:ending
      filename = filenames{K};
      [~, basenames{K}, ~] = fileparts(filename);
       try
        all_data_cell{K} = imread(filename);
        A = (all_data_cell{K});
        A = flipud(medfilt2(rescale(A(:,:,1))));

        text_str = ['Time:' num2str(t_interval(K),'%0.1f') ' h'];

        position = [50 50]; 
        box_color = {'blue'};

        RGB = insertText(A,position,text_str,'FontSize',50,'BoxColor',box_color,'BoxOpacity',0.1,'TextColor','white');

        fig = figure(1)
        imshow(RGB)
        
        F(K-ini+1) = getframe(fig);

   catch ME
        fprintf('Warning: problem importing file "%s"\n', filename);
      end
      K
    end

   
v = VideoWriter(videoname); %% I think you can change the format e.g. mp4
v.FrameRate = 2;
open(v)
writeVideo(v,F)
close(v)

%%
for d = 1:length(D)-2 %!!!
    
    clear avspecint 
    timemultiple = d;
    cd(scriptpath)
    wavref = wav_ini:step:wav_fin-step*2;
    wavint = interp1(1:length(wavref),wavref,linspace(1,length(wavref),1000));
    wavref = interp1(1:length(wavint),wavint,linspace(1,length(wavint),length(wavref)));
    peak = 1;
    colaxis = [wav_ini wav_fin];
    %% Data importing

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

    for n = 1:length(A)
        eval(['im' num2str(n) '=importdata(string(Aname(n)));']);
        fprintf(format,n,length(A))
    end
    clear n

    % Image dimensions
    dimensions = size(im1);
    xsize = dimensions(1);
    ysize = dimensions(2);

    % Image stack:
    %imstack = zeros(xsize,ysize,length(A));
    for n = 1:length(A)
        im = eval(['im' num2str(n)]);
        imstack(:,:,n) = im(:,:,1);
      
    end

    clear n
    format = 'Data importing finished \n';
    fprintf(format);


    %% Main loop to locate resonance at every pixel


    %f = fit(month,pressure,'smoothingspline');
    % main loop to find resonance
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

    %% Display the resonance map

    figure(1);
    pcolor(map)
    shading interp
    caxis([colaxis(1) colaxis(2)]);
    colormap(autumn);
    colorbar
    title('Resonance map','FontSize',12)
    axis equal
    set(gca,'xticklabel','','yticklabel','')

    format = 'Plotting finished \n';
    fprintf(format);

    %% ROI selection
    if d == 1
        format = 'Select ROI for average resonance check \n';
        fprintf(format);
        [x,y] = ginput(2);
    end
    x = round(x); y = round(y);

    region = imstack(y(1):y(2),x(1):x(2),1:end);
    mapregion = map(y(1):y(2),x(1):x(2));
    ar = [x(1) x(2);y(2) y(2)];
    regsize = size(region);

    fig1 = figure(1);
    pcolor(wiener2(map,[2 2]))
    shading interp
    caxis([colaxis(1) colaxis(2)]);
    colormap(autumn);
    colorbar
    title('Resonance map','FontSize',12)
    set(gca,'xticklabel','','yticklabel','')
    hold on
    area(ar(1,:),ar(2,:),y(1),'FaceColor','none','EdgeColor','k','LineWidth',2)
    hold off
    axis tight equal
    drawnow


    %% ROI interpolation
    temp = reshape(region,regsize(1)*regsize(2),length(wavref));
    avspec = mean(temp,1);
    clear temp


    %% Average spectrum calculation, levelling, and plotting from interpolation
    avspecint = interp1(wavref,avspec,wavint);
    levellingint = linspace(avspecint(1),avspecint(end),length(avspecint));
    cd(scriptpath)
    [specfit,gof] = fanofit(wavref,avspec);
    fitted = feval(specfit,wavref);

    fig2 = figure(2)
    p = plot(wavref,avspec,'or',wavref,fitted,'-k');
    set(p,'MarkerSize',2,'LineWidth',1);
    xlabel('Wavelength (nm)')
    ylabel('Reflectance [a.u.]')
    set(gca,'Fontsize',15)

    format = 'Average spectrum calculation finished \n';
    fprintf(format);


    %
    %% Average spectrum analysis
    [~,in] = max(fitted);
    reswav(d) = wavint(in);

end

%% Saving

cd(savepath)
save(mapfilename,'mapstore')

%%

    %% select your map
cd(savepath)
load(mapfilename) 
A = (all_data_cell{end});
     
format = 'Select ROIs manually \n';
fprintf(format);
for n = 1:cellnum
  [A_cells{n}, rect{n}]  = imcrop(medfilt2((rescale(A)))); 
end
cd(savepath)
save('rect.mat','rect')   
cd(HyS_path)

%%

for n = 1:cellnum
    for K = ini:ending
      filename = filenames{K};
      [~, basenames{K}, ~] = fileparts(filename);
      try
        all_data_cell{K} = imread(filename);
        A = (all_data_cell{K});
        A = imcrop(A,rect{n}); 
        A = (medfilt2(rescale(A(:,:,1))));


      catch ME
        fprintf('Warning: problem importing file "%s"\n', filename);
      end
      K
    


        %% calculating average resonance wavelength of each ROI

        map1 =mapstore(:,:,K);
        refmap = imcrop(map1,rect{1}); 
        map1 = imcrop(map1,rect{n}); 
          
        avref1(K) = mean(mean(mean(refmap(:,1))));
        avref2(K) = mean(mean(mean(refmap(1,:))));
        avref(K) = mean([avref1(K) avref2(K)]);
        avref_norm(K) = avref(K)-avref(1);


        av1(K) = mean(mean(mean(map1(:,1))));
        av2(K) = mean(mean(mean(map1(1,:))));
        av(K) = mean(mean(map1));
        av_norm(K) = av(K)-av(1);

        map1_wiener = wiener2(wiener2(wiener2(map1,[20 20])));
        map1_norm = (map1_wiener-av(K));


      
        %% imshow
        fig3 = figure(3)
        sub(1) = subplot(1,2,1)
        imshow(map1_norm)
        title(['resmap of picowell ',sprintf('%d',n)])
        shading interp
        caxis([-1 1]);

        colormap(jet(128));
        c = colorbar;
        c.Label.String = '\Delta\lambda [nm]';
        set(gca,'xticklabel','','yticklabel','')
        set(gca,'Fontsize',10)

        sub(2) = subplot(1,2,2)
        imshow(A)
        title('Phase contrast')
        colormap(gray);



        set(gca,'xticklabel','','yticklabel','')
        set(gca,'Fontsize',10)
        %   

        colormap(sub(1),jet(128))
        colormap(sub(2),gray)



        
    end
    
   
    cd(savepath)
    save(strcat('av_',num2str(n)),'av_norm')

    av_norm = (smooth(av_norm));
    hist(n) = (av_norm(fin));
    fig4 = figure(4)
    plot(t_interval,av_norm)
    hold on
    xlabel('Time after activation [h]')
    ylabel('Resonance wavelength [nm]')
    set(gca,'Fontsize',15)

end
cd(savepath)
 saveas(gca,res_over_t_name)  

%%
fig5 = figure(5)
histogram(hist,'BinWidth',0.1)
legend(legend_name)
xlabel('Resonance shift [nm]')
ylabel('#cells')
set(gca,'Fontsize',15)
%xlim([-0.8 0.8])
%ylim([0 13])
saveas(gca,histogram_name)

cd(savepath)
save(histogram_workspace,'hist')  
%%
fig6 = figure(6)
boxplot(hist)
title(legend_name)
saveas(gca,boxplot_name)
ylabel('Resonance shift [nm]')
set(gca,'Fontsize',15)
%%
close(fig1)
close(fig2)
close(fig3)