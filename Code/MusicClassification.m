clc 
clearvars -except Specs U S V BandLabels
close all


% VectorizingSpectrograms.m

%  load AllSongdata.mat

BandLabels(1:250,1) = repmat({'Deadmau5'},250,1);
BandLabels(251:500,1) = repmat({'Flume'},250,1);
BandLabels(501:750,1) = repmat({'Kaskade'},250,1);
BandLabels(751:1000) = repmat({'Heiroglyphics'},250,1);
BandLabels(1001:1250) = repmat({'RunTheJewels'},250,1);
BandLabels(1251:1500) = repmat({'Jurassic5'},250,1);
BandLabels(1501:1750) = repmat({'STRFKR'},250,1);
BandLabels(1751:2000) = repmat({'Phantogram'},250,1);
BandLabels(2001:2250) = repmat({'UMO'},250,1);

GenreLabels(1:750,1) = repmat({'Electronic'},750,1);
GenreLabels(751:1500,1) = repmat({'HipHop'},750,1);
GenreLabels(1501:2250,1) = repmat({'IndieRock'},750,1);
ratio=0.5;
k=250;
nrand=k*ratio;
ntest=k-nrand;
randSong = randperm(k,nrand);
testSong = setdiff(1:250,randSong);

kg=750;
nrandg=kg*ratio;
ntestg=kg-nrandg;
randGen = randperm(kg,nrandg);
testGen = setdiff(1:750,randGen);

SetDataLim
randElec=randGen;
testElec=setdiff(1:750,randElec);
randHipHop=750+randGen;
testHipHop=setdiff(751:1500,randHipHop);
randIndie=1500+randGen;
testIndie=setdiff(1501:2250,randIndie);
Genres={'Electronic','HipHop','IndieRock'};
Electronic={'Deadmau5','Flume','Kaskade'};
HipHop={'Heiroglyphics','RunTheJewels','Jurassic5'};
IndieRock={'STRFKR','Phantogram','UMO'};

% [U,S,V]=svd(Specs-(mean(Specs,1).*ones(size(Specs,1),1)),'econ');
% 
% case 1
% CMK="Cross-Genre Artist Classification";
% Bands={'Deadmau5','Heiroglyphics','Phantogram'}; %% case 1
% Labels = BandLabels([randDm5,randHeiro,randPhant]);
% X=(V(:,[randDm5,randHeiro,randPhant]).^(1/8))';
% test=(V(:,[testDm5,testHeiro,testPhant]).^(1/8))';
% Reality=BandLabels([testDm5,testHeiro,testPhant]);
% figtitl='Artists of Different Genres';
% testU=(U(:,[testDm5,testHeiro,testPhant]).^(2))';
% testS=(S([testDm5,testHeiro,testPhant],[testDm5,testHeiro,testPhant]).^(2))';
% trnS=(S([randDm5,randHeiro,randPhant],[randDm5,randHeiro,randPhant]).^(2))';

% % case 2 a
% CMK="Hip Hop Artist Classification";
% Bands=HipHop; %% case 2
% Labels = BandLabels([randHeiro,randRTJ,randJ5]);
% X=(V(:,[randHeiro,randRTJ,randJ5]).^(1/5))';
% test=(V(:,[testHeiro,testRTJ,testJ5]).^(1/5))';
% Reality=BandLabels([testHeiro,testRTJ,testJ5]);
% figtitl='HipHop Artists';
% testU=(U(:,[testHeiro,testRTJ,testJ5]).^(2))';
% testS=(S([testHeiro,testRTJ,testJ5],[testHeiro,testRTJ,testJ5]).^(2))';
% trnS=(S([randHeiro,randRTJ,randJ5],[randHeiro,randRTJ,randJ5]).^(2))';

% % case 2 b
% CMK="Electronic Artist Classification";
% Bands=Electronic; %% case 2
% Labels = BandLabels([randDm5,randFlm,randKask]);
% X=(V(:,[randDm5,randFlm,randKask]).^(1/2))';
% test=(V(:,[testDm5,testFlm,testKask]).^(1/2))';
% Reality=BandLabels([testDm5,testFlm,testKask]);
% figtitl='Electronic Artists';
% testU=(U(:,[testDm5,testFlm,testKask]).^(2))';
% testS=(S([testDm5,testFlm,testKask],[testDm5,testFlm,testKask]).^(2))';
% trnS=(S([randDm5,randFlm,randKask],[randDm5,randFlm,randKask]).^(2))';

% % case 2 c
% CMK="Indie Rock Artist Classification";
% Bands=IndieRock; %% case 2
% Labels = BandLabels([randSTRF,randPhant,randUMO]);
% X=(V(:,[randSTRF,randPhant,randUMO]).^(1/8))';
% test=(V(:,[testSTRF,testPhant,testUMO]).^(1/8))';
% Reality=BandLabels([testSTRF,testPhant,testUMO]);
% figtitl='Indie Rock Artists';
% testU=(U(:,[testSTRF,testPhant,testUMO]).^(2))';
% testS=(S([testSTRF,testPhant,testUMO],[testSTRF,testPhant,testUMO]).^(2))';
% trnS=(S([randSTRF,randPhant,randUMO],[randSTRF,randPhant,randUMO]).^(2))';

% % case 3
% CMK="Genre Classification";
% Bands=Genres; %% case 2
% Labels = GenreLabels([randElec,randHipHop,randIndie]);
% X=(V(:,[randElec,randHipHop,randIndie]).^(1/3))';
% test=(V(:,[testElec,testHipHop,testIndie]).^(1/3))';
% Reality=GenreLabels([testElec,testHipHop,testIndie]);
% figtitl='Classifying by Genre';
% testS=(S([testElec,testHipHop,testIndie],[testElec,testHipHop,testIndie]).^(2))';
% trnS=(S([randElec,randHipHop,randIndie],[randElec,randHipHop,randIndie]).^(2))';

BandClassify= fitcecoc(real(X),Labels,...
    'ClassNames',Bands,...
    'Learners','svm','FitPosterior',true);
test=real(test);
[label,score,cost,PosteriorRegion] = predict(BandClassify,test);
[label2,~,~,Posterior] = resubPredict(BandClassify,'Verbose',2);

Accuracy=0;
for I=1:size(label,1)
    tf = isequal(label(I), Reality(I));
    Accuracy=Accuracy+tf;
end
ErrorSVM=((size(test,1)-Accuracy)/size(test,1))*100
cm = confusionchart(Reality,label,'RowSummary','row-normalized','Normalization','absolute');
set(gcf,'Position',[535,528,900,150])
title(CMK+": ECOC Model - % of Dataset Utilized for Training/Testing = "+string(ratio*100)+"% : "+string((1-ratio)*100)+"%" )
print(CMK+string(ratio*100)+"%testset.jpg",'-djpeg')
