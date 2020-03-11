clc
clear all
close all

% SortTheFaces
load cropped1.mat
k=size(faces,2);
randface = randperm(k,1);

X=faces;

DataRank=rank(X); % Computing rank
[m,n]=size(X); % compute data size
mn=mean(X,1); % compute mean for each column
X=X-repmat(mn',1,size(X,1))'; % subtract mean from X
[u,s,v]=svd(X/sqrt(n-1),'econ'); % perform the SVD
semilogx(1:size(s,1),diag(s).^2,'o');
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

indd=mainfeat(randface,:);
upl=u(:,indd);

figure(2)
for j=1:4;
    subplot(1,4,j)
    imshow(mat2gray(reshape(upl(:,j),192,168))); drawnow
end

for fct=1:fct;
    figure(3)
    subplot(3,4,[5,6,9,10]-[4,4,4,4])
    
    LoRank=u(:,indd(1:fct))*s(indd(1:fct),indd(1:fct))*v(:,indd(1:fct)).';
    
    imshow(mat2gray(reshape(LoRank(:,randface),192,168)))
    
    title(string(fct)+" Feature Reconstruction")
    
    subplot(3,4,[7,8,11,12]-[4,4,4,4])
    
    imshow(mat2gray(reshape(faces(:,randface),192,168)))
    title('Original Image')
    disp(string(fct)+" Mode Reconstruction Error")
    
    Error(fct)=immse(faces(:,randface),LoRank(:,randface));
 
    subplot(3,4,[9,10,11,12])
    semilogy(1:fct,Error,'o')
    xlabel('Number of Modes')
    ylabel('Mean Squared Error')
    title(sprintf("MSE = %0.1f%", Error))
    
end

