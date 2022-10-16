clc; clear all; close all;
% cross  corpus
list1={'iemo_base_ster_AHNS-name.arff',...
       'iemo_base_AHNS-name.arff',...
       'iemo_ster_AHNS-name.arff'};
list2={'5mix_base_ster-name.arff',...
       '5mix_base-name.arff',...
       '5mix_ster-name.arff'};
dir1='E:\PhD\articles\Arabian\Table9\';
len1=size(list1,2);
AAA=[];
kk4=10;
kk5=1;
normtype=2; % normalisation type of data
B3=[];
for kk5=1:1
B2=[];
for kk6=1:1
B1=[];

F2=[];
G2=[];
H2=[];
AAA2=[];

for kk2=1:len1
%%%%% train data    
name1=list1{kk2};

[dir1 name1]
w1=loadARFF([dir1 name1]);
[mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(w1);
data=mdata(:,1:end-1);
labels=mdata(:,end);
labels=labels+1;
% return
% normtype=2; % normalisation type of data
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

%%%%% test data
name1=list2{kk2};

[dir1 name1]
w1=loadARFF([dir1 name1]);
[mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(w1);
data=mdata(:,1:end-1);
labels=mdata(:,end);
labels=labels+1;

% zero out small values
% b=data;
% cc=[];
% for i=1:size(b,2)
% c=b(:,i);
% max1=max(c);
% c(c<max1/100)=0;
% b(:,i)=c;
% end
% data=b;

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
H1=[];


for kk=1:10
ini = clock; 
nword=size(C,1);
Inputs = size(TrainData,2);    % # of variables as input
Outputs = nword;               % # of variables as output
hidden       = 125;% ceil(0+kk5*5); % 125
opts.MaxIter = 1000;% 0+kk6*5;% return
% nodes must be set correctly
nodes = [Inputs hidden Outputs]; % [#inputs #hidden #outputs]
bbdbn = randDBN(nodes, 'BBDBN'); % Bernoulli-Bernoulli RBMs
nrbm = numel(bbdbn.rbm);
% hyper-parameters

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
opts.StepRatio = 2.5/1000;
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
s2=sum(s1(:) == 1);
s3=sum(s1(:) == 0);
ss=s2/(s2+s3);
A=[A;ss]
H1=[H1; hidden opts.MaxIter];
end
AA=sum(A)/size(A,1);
AAA2=[AAA2 A];
B1=[AAA2; mean(AAA2)]

end
% F2=[F2; hidden opts.MaxIter]
% Z2=zeros(1,size(B1,2));
% H2=[H2;Z2;Z2;H1];
% G2=[G2;Z2;B1]
B2=[B2 B1]

end

B3=[B3 B2]

end