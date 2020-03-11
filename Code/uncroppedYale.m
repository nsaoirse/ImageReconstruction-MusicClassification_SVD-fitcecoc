clc
clear all
close all

% it might be cool to look at how well an SVD can extract someones face
% from all the other cases 

% % % % Yale Faces B
% % Download two data sets (ORIGINAL IMAGE and CROPPED IMAGES)
% % Your job is to perform an SVD analysis of these data sets. Please start with the cropped images and
% % perform the following analysis.

% 1. Do an SVD analysis of the images (where each image is reshaped into a column vector and
% each column is a new image).

% 2. What is the interpretation of the U, ? and V matrices?

% 3. What does the singular value spectrum look like and how many modes are necessary for good
% image reconstructions? (i.e. what is the rank r of the face space?)

% 4. compare the difference between the cropped (and aligned) versus uncropped images.

% This is an exploratory homework. So play around with the data and make sure to plot the different
% things like the modes and singular value spectrum. Good luck, and have fun.

% SortTheFaces
load uncroppedfaces.mat
k=size(faces,2);
randface = randperm(k,1);

X=faces;

DataRank=rank(X); % Computing rank
[m,n]=size(X); % compute data size
mn=mean(X,1); % compute mean for each column
X=X-repmat(mn',1,size(X,1))'; % subtract mean from X
[u,s,v]=svd(X/sqrt(n-1),'econ'); % perform the SVD

semilogx(1:size(s,1),diag(s).^2);
xlabel('Mode')
ylabel('Energy')
title('Power by Orthogonal Mode')


%% Lo-Rank
nfeat=size(v,1);
fct=26;
figure(2)

for f=1:size(v,2);
    [val ind] = sort(s(:,f),'descend');
    mainfeat(f,:)=ind(1:fct);
end
% for i=1:nfeat;
%     imshow(mat2gray(reshape(u(:,i),243,320))); drawnow
%     pause(1)
% end
indd=mainfeat(randface,:);
upl=u(:,indd);

figure(2)
for j=1:4;
    subplot(1,4,j)
imshow(mat2gray(reshape(upl(:,j),243,320))); drawnow
end

for fct=1:fct;
    figure(3)
    subplot(3,4,[5,6,9,10]-[4,4,4,4])
    
    LoRank=u(:,indd(1:fct))*s(indd(1:fct),indd(1:fct))*v(:,indd(1:fct)).';
    
    imshow(mat2gray(reshape(LoRank(:,randface),243,320)))
    
    title(string(fct)+" Feature Image Reconstruction")
    
    subplot(3,4,[7,8,11,12]-[4,4,4,4])
    
    imshow(mat2gray(reshape(faces(:,randface),243,320)))
    title('Original Image')
    disp(string(fct)+" Mode Reconstruction Error")
    
    Error(fct)=immse(reshape(faces(:,randface),243,320),reshape(LoRank(:,randface),243,320));
 
    subplot(3,4,[9,10,11,12])
    semilogy(1:fct,Error,'o')
    xlabel('Number of Modes')
    ylabel('Mean Squared Error')
    title(sprintf("MSE = %0.1f%", Error))
    
end