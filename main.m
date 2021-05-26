%------------------------------- 1 -------------------------------
Data = csvread('Indian Liver Patient Dataset (ILPD).csv');

%Κανικοποίηση Δεδομένων
normData = -1 + (Data(:,1:10) - min(Data(:)))*2/(max(Data(:) - min(Data(:))));
temp = num2str(Data(:,11));
categories_cell = num2cell(temp);

X = normData;
Y = categories_cell;

%------------------------------- 3 -------------------------------

%---------------Αξιολόγηση ΝΒ---------------
indices = crossvalind('Kfold',Y,5);
NB_cp = classperf(Y);
for i = 1:5
    test = (indices == i);
    train = ~test;
    NB_Model = fitcnb(X(train,:),Y(train));
    NB_predictions = predict(NB_Model,X(test,:));
    NB_cp = classperf(NB_cp,NB_predictions,test);
end
g_mean = sqrt(NB_cp.Sensitivity*NB_cp.Specificity);
NB_Metrics = [NB_cp.Sensitivity NB_cp.Specificity g_mean];

%------------------------------- 4 -------------------------------

%---------------Βελτιστοποίηση SVM---------------
c = cvpartition(583,'KFold',5);

%----Βέλτιστο C----
%Δεν μπόρεσα να κανω διαδοχική αναζήτηση με βήμα
box = optimizableVariable('box',[1,200],'Type','integer','Transform','log');
 
minfn = @(z)kfoldLoss(fitcsvm(X,Y,'CVPartition',c,...
    'KernelFunction','linear','BoxConstraint',z.box,...
    'KernelScale','auto'));
 
results = bayesopt(minfn,box,'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus');  

z(1) = results.XAtMinObjective.box;
opt_box = z(1);

%----Βέλτιστο γ----

sigma = optimizableVariable('sigma',[1,10],'Transform','log');
 
minfn = @(z)kfoldLoss(fitcsvm(X,Y,'CVPartition',c,...
     'KernelFunction','rbf','BoxConstraint',opt_box,...
     'KernelScale',z.sigma));
 
results = bayesopt(minfn,sigma,'IsObjectiveDeterministic',true,...
     'AcquisitionFunctionName','expected-improvement-plus');
 
z(2) = results.XAtMinObjective.sigma;
opt_s = z(2);


%---------------Αξιολόγηση SVM---------------

indices = crossvalind('Kfold',Y,5);
SVM_cp = classperf(Y);
for i = 1:5
    test = (indices == i); 
    train = ~test;
    SVM_Model = fitcsvm(X(train,:),Y(train),'KernelFunction','rbf','KernelScale',z(2),'BoxConstraint',z(1));
    SVM_predictions = predict(SVM_Model, X(test,:));
    classperf(SVM_cp, SVM_predictions, test);

end
g_mean = sqrt(SVM_cp.Sensitivity*SVM_cp.Specificity);
SVM_Metrics = [SVM_cp.Sensitivity SVM_cp.Specificity g_mean];



%---------------Βελτιστοποίηση ΚΝΝ---------------
k = optimizableVariable('NumNeighbors',[3,15],'Type','integer','Transform','log');

rng(1)
KNN_Model = fitcknn(X,Y,'OptimizeHyperparameters',k,...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus','CVPartition',c));

opt_k = KNN_Model.NumNeighbors;
%---------------Αξιολόγηση ΚΝΝ---------------
indices = crossvalind('Kfold',Y,5);
KNN_cp = classperf(Y);
for i = 1:5
    test = (indices == i); 
    train = ~test;
    KNN_predictions = predict(KNN_Model, X(test,:));
    classperf(KNN_cp, KNN_predictions, test);
end
g_mean = sqrt(KNN_cp.Sensitivity*KNN_cp.Specificity);
KNN_Metrics = [KNN_cp.Sensitivity KNN_cp.Specificity g_mean];


disp(NB_Metrics);
disp(KNN_Metrics);
disp(SVM_Metrics);


%------------------------------- 5 -------------------------------

mean_y = 0;
for i = 1:583
    mean_y = mean_y + Data(i,11)/583;
end

for j = 1:10

    mean_x = 0;
    for i = 1:583
        mean_x = mean_x + Data(i,j)/583;
    end
    
    s = [0 0 0];
    for i = 1:583
        x = Data(i,j);
        y = Data(i,11);
        
        s(1) = s(1) + (x-mean_x)*(y-mean_y);
        s(2) = s(2) +  (x-mean_x)^2;
        s(3) = s(3) +  (y-mean_y)^2;
    end
    
    r(j,1) = j;
    r(j,2) = s(1)/(sqrt(s(2))*sqrt(s(3)));
    
end

r = sortrows(r,2,'descend');

%Το ερώτημα ζητάει τις 5 μεταβλητές με τον μεγαλύτερο θετικο συντελεστή
%αλλα μόνο 4 είχαν θετικο συντελεστή
%οπότε χρησιμοποιώ όλες όσες έχουν θετικό συντελεστή

j = 1;
for i = 1:10
    if r(i,2)>0
        column = r(i,1);
        newData(:,j) = Data(:,column);
        j = j+1;
    end
end

newX = newData;

%---------------Αξιολόγηση ΝΒ---------------

indices = crossvalind('Kfold',Y,5);
NB_cp = classperf(Y);
for i = 1:5
    test = (indices == i);
    train = ~test;
    NB_Model = fitcnb(newX(train,:),Y(train));
    NB_predictions = predict(NB_Model,newX(test,:));
    NB_cp = classperf(NB_cp,NB_predictions,test);
end
g_mean = sqrt(NB_cp.Sensitivity*NB_cp.Specificity);
newNB_Metrics = [NB_cp.Sensitivity NB_cp.Specificity g_mean];



%---------------Αξιολόγηση SVM---------------

indices = crossvalind('Kfold',Y,5);
SVM_cp = classperf(Y);
for i = 1:5
    test = (indices == i); 
    train = ~test;
    SVM_Model = fitcsvm(newX(train,:),Y(train),'KernelFunction','rbf','KernelScale',opt_s,'BoxConstraint',opt_box);
    SVM_predictions = predict(SVM_Model, newX(test,:));
    classperf(SVM_cp, SVM_predictions, test);

end
g_mean = sqrt(SVM_cp.Sensitivity*SVM_cp.Specificity);
newSVM_Metrics = [SVM_cp.Sensitivity SVM_cp.Specificity g_mean];



%---------------Βελτιστοποίηση ΚΝΝ---------------
indices = crossvalind('Kfold',Y,5);
KNN_cp = classperf(Y);
for i = 1:5
    test = (indices == i); 
    train = ~test;
    KNN_Model = fitcknn(newX(train,:),Y(train), 'NumNeighbors', opt_k);
    KNN_predictions = predict(KNN_Model, newX(test,:));
    classperf(KNN_cp, KNN_predictions, test);
end
g_mean = sqrt(KNN_cp.Sensitivity*KNN_cp.Specificity);
newKNN_Metrics = [KNN_cp.Sensitivity KNN_cp.Specificity g_mean];


disp(newNB_Metrics);
disp(newKNN_Metrics);
disp(newSVM_Metrics);