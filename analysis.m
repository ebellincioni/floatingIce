% test on reading image and finding contour
I = imread('/Users/edoardo/Library/CloudStorage/OneDrive-UniversityofTwente/floating/2023-09-12/DSC_2818.NEF');
I2 = mat2gray(I);
figure()
imshow(I2(1500:3500,3500:5500));
axis on

roi = drawassisted;
a = roi.Position;
%% test on plotting contour
figure()
scatter(a(:,1),a(:,2))
pbaspect([1 1 1])
grid on
%% reading all images
addpath('/Users/edoardo/Library/CloudStorage/OneDrive-UniversityofTwente/floating/github_repo');
readraw;
directory = '/Users/edoardo/Library/CloudStorage/OneDrive-UniversityofTwente/floating/2023-10-19/2nd/';
imagefiles = dir(strcat(directory,'DSC*'));      
nfiles = length(imagefiles);    % Number of files found
contours = {};
for ii=1:7:nfiles
    currentfilename = imagefiles(ii).name;
    %currentimage = imread(strcat(directory,currentfilename));
    I = imread(strcat(directory,currentfilename));
    I2 = mat2gray(I);
    f = figure();
    f.Position(3:4) = [2000,3200];
   
    % imshow(I2(1000:3500,2000:6000)); % y,x
    imshow(I2);
    title(currentfilename);
    axis on
    
    roi = drawassisted;
    %contours = [contours;roi.Position];
    contours = roi.Position;
    set(gcf,'position',[1000,1000,2000,3200])
    csvwrite(directory+"contours/"+sprintf('%04d',ii)+".csv",contours)
    close
end 
%% show contours
figure()
for i=1:length(contours)
    %disp([contours{i}(:,1),contours{i}(:,2)])
    scatter(contours{i}(:,1),contours{i}(:,2))
    hold on
end
figure()
scatter(contours{1}(:,1),contours{1}(:,2))
%% for 
for i=1:length(contours)
    %csvwrite(transpose(contours(i)),"/Users/edoardo/Library/CloudStorage/OneDrive-UniversityofTwente/floating/2023-09-12/2nd/contours/"+num2str(i)+".csv")
    csvwrite("/Users/edoardo/Library/CloudStorage/OneDrive-UniversityofTwente/floating/2023-09-12/2nd/contours/"+sprintf('%02d',i)+".csv",transpose(contours(i)))
end