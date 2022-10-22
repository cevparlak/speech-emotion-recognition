clc; clear all; close all;
list1={'emobase_A_N_H_S.mat','emostarbase.mat',      'emodb_ster_A_N_H_S.mat', 'emostar_ster.mat',        'emobase_ster_A_N_H_S.mat',  'emostarbase_ster.mat'};
list2={'emostarbase.mat',    'emobase_A_N_H_S.mat',  'emostar_ster.mat',       'emodb_ster_A_N_H_S.mat',  'emostarbase_ster.mat',      'emobase_ster_A_N_H_S.mat'};
% list1={'emostarbase.mat',          'emostarbase_ster.mat'};
% list2={'emobase_A_N_H_S.mat',      'emobase_ster_A_N_H_S.mat'};
list1={'emostarbase_ster.mat'};
list2={'emobase_ster_A_N_H_S.mat'};
len1=size(list1,2);
AAA=[];
for kk2=1:len1
name1=list1{kk2}
load (name1)
% return
% filename='emostarbase.mat';
% d=TrainData;
labels=data(:,end);
data=data(:,1:end-1);
% return
rng('default');

normtype=2; % normalisation type of data
data=normalise_data(data,normtype);
data(isnan(data))=0;
% data( :, ~any(data,1) ) = [];  % remove zero columns

[C,ia,ic] = unique(labels);
nclasses=size(C,1);
a_counts = accumarray(ic,1);
value_counts = [C, a_counts];

TrainData=data;
TrainLabels=labels;
TrainLabels2=labels;

name2=list2{kk2}
load (name2)
% filename='emostarbase.mat';
% d=data;
labels=data(:,end);
data=data(:,1:end-1);

data=normalise_data(data,normtype);
data(isnan(data))=0;
% data( :, ~any(data,1) ) = [];  % remove zero columns

TestData=data;
TestLabels=labels;
TestLabels2=labels;

% flip train and test
% t1=TrainData;
% t2=TrainLabels;
% TrainData=TestData;
% TrainLabels=TestLabels;
% TrainLabels2=TestLabels;
% TestData=t1;
% TestLabels=t2;
% TestLabels2=t2;

% return
% binarize labels
lbl1=[];
for i1=1:size(TrainLabels,1)
l1=zeros(1,nclasses);
l2=TrainLabels(i1,1);
l1(1,l2)=1;
    lbl1=[lbl1; l1];
end
TrainLabels=fliplr(lbl1);

lbl1=[];
for i1=1:size(TestLabels,1)
l1=zeros(1,nclasses);
l2=TestLabels(i1,1);
l1(1,l2)=1;
    lbl1=[lbl1; l1];
end
TestLabels=fliplr(lbl1);
% return
A=[];
for kk=1:10
ini = clock; 
nword=size(C,1);
Inputs = size(TrainData,2);    % # of variables as input
Outputs = nword;               % # of variables as output
hidden = 125;
% return
nodes = [Inputs hidden Outputs]; % [#inputs #hidden #outputs]
bbdbn = randDBN(nodes, 'BBDBN'); % Bernoulli-Bernoulli RBMs
nrbm = numel(bbdbn.rbm);
% hyper-parameters
opts.MaxIter = 1000;
opts.BatchSize = 25;
opts.Verbose = true;
opts.InitialMomentumIter = opts.MaxIter/5;
opts.InitialMomentum = 0.5;
opts.FinalMomentum = 0.9;
opts.WeightCost = 0.0002;
opts.DropOutRate=0.; % must be > 0.9
opts.SparseLambda=0.;
opts.SparseQ=0.;
% for GBDBN stepratio must be 1 or 2 order higher than BBDBN
opts.StepRatio = 25/1000;
opts.object = 'CrossEntropy';
% Learning stage
fprintf( 'Pretraining DBN...\n' );
opts.Layer = nrbm-1;
bbdbn = pretrainDBN(bbdbn, TrainData, opts);
bbdbn= SetLinearMapping(bbdbn, TrainData, TrainLabels);
opts.Layer = 0;
fprintf( 'Training DBN...\n' );
bbdbn = trainDBN(bbdbn, TrainData, TrainLabels, opts);

theend = clock; 
    traintime1=etime(theend,ini);
    fprintf('\nTraining time: %f\n',traintime1);   
    
fprintf( 'Testing...\n' );
ini = clock; 
out = v2h( bbdbn, TestData );
theend = clock; 
    testtime1=etime(theend,ini);
    fprintf('\nTesting time: %f\n',testtime1);   

% reorganize out labels
B=out;
BB=[];
for i=1:size(out,1)
dd1=out(i,:);
%A = rand(1,500) ;
id = find(dd1==max(dd1)) ;
dd1=zeros(1,nclasses);
dd1(id) = 1;
% A=fliplr(A);
B(i,:)=dd1;
BB(i,1)=nclasses+1-id(1);
% return
end

s1=(BB==TestLabels2);
s2=sum(s1(:) == 1)
s3=sum(s1(:) == 0)
ss=s2/(s2+s3)
A=[A;ss]
end
AA=sum(A)/size(A,1);
AAA=[AAA A]
end
B1=[AAA; mean(AAA)]
