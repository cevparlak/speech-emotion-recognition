clc; clear all; close all;
list1={'iemo_base_AHNS-name.arff',...
       'iemo_base_ster_AHNS-name.arff',...
       'iemo_ster_AHNS-name.arff'};
list1={'iemo_ster_AHNS-name.arff'};
         
dir1='E:\PhD\articles\Arabian\Table9\';
len1=size(list1,2);
AAA=[];
kk4=10;
B1=[];
B2=[];
B3=[];
for kk5=1:5
for kk6=1:5
for kk2=1:len1
    
    kk3=kk2;
    kk4=kk4+1;
name1=list1{kk3}
w1=loadARFF([dir1 list1{kk3}]);
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
% return
normtype=2; % normalisation type of data
data=normalise_data(data,normtype);
data(isnan(data))=0;
data(:,~any(data,1))=[]; % remove zero columns
data=[data labels];
% return
rng('default');
% % Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;
% return
% Separate to training and test data
TrainData  = data(~idx,1:end-1);
TrainLabels= data(~idx,end);
TrainLabels2= data(~idx,end);
TestData   = data(idx,1:end-1);
TestLabels = data(idx,end);
TestLabels2 = data(idx,end);
[C,ia,ic] = unique(TrainLabels);
nclasses=size(C,1);
a_counts = accumarray(ic,1);
value_counts = [C, a_counts];
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
Inputs = size(TrainData,2);       % # of variables as input
Outputs = nclasses;               % # of variables as output
hidden = ceil(15+kk5*20); % 125
% return
nodes = [Inputs hidden Outputs]; % [#inputs #hidden #outputs]
bbdbn = randDBN(nodes, 'BBDBN');
nrbm = numel(bbdbn.rbm);
% hyper-parameters
opts.MaxIter = kk6*200;
opts.BatchSize = 25;
opts.Verbose = true;
opts.InitialMomentumIter = opts.MaxIter/5;
opts.InitialMomentum = 0.5;
opts.FinalMomentum = 0.9;
opts.WeightCost = 0.0002;
opts.DropOutRate=0.; % must be > 0.9
opts.SparseLambda=0.;
opts.SparseQ=0.;
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
id = find(dd1==max(dd1));
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
A=[A;mean(A)];
A=[hidden;opts.MaxIter;opts.StepRatio;A];
AAA=[AAA A];
B1=AAA;
A=[];
AAA=[];

end
B2=[B2 B1]
B1=[];

end

B3=[B3; B2]
B2=[];
end
