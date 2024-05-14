load('Generdt_zero200.mat');
load('GenerImp200.mat');
load('Generpower200.mat');
dt = 0.01;
GenerImp200Info_idxfirst = zeros(1,size(GenerImp200,2));
GenerImp200Info_idxlast = zeros(1,size(GenerImp200,2));
GenerImp200Info_dt = zeros(1,size(GenerImp200,2));
GenerImp200Info_power = zeros(1,size(GenerImp200,2));

% Calculate the duration of each generated impulse
for i = 1:size(GenerImp200,2)
    % Identify the beginning point 
    GenerImp200Info_idxfirst(1,i) = find(GenerImp200(:,i)-1<0.05,1,'first');
    % Identify the ending point 
    [~,Info_peak_idxlast_temp] = findpeaks(GenerImp200(:,i),'SortStr','descend'); % Peak value identification
    Info_idxlast_temp = zeros(1,length(GenerImp200(:,i)));
    k = 0;
    for j = 1:length(GenerImp200(:,i))-1
        if GenerImp200(j,i)-1>0.05 && GenerImp200(j+1,i)-1<0.05
            k = k + 1;
            Info_idxlast_temp(k) = j;
        end
    end
    % If there is no intersection point
    if max(Info_idxlast_temp) <= Info_peak_idxlast_temp(1)
        continue
    end
    idxtemp = find(Info_idxlast_temp - Info_peak_idxlast_temp(1)>0,1,'first');
    GenerImp200Info_idxlast(1,i) = Info_idxlast_temp(idxtemp);
end

% Calculate the parameters of the generated sample
for i = 1:size(GenerImp200,2)
    GenerImp200Info_dt(1,i) = (GenerImp200Info_idxlast(1,i)- GenerImp200Info_idxfirst(1,i))*0.01;
end

for i = 1:size(GenerImp200,2)
    x = (0:(GenerImp200Info_idxlast(1,i)- GenerImp200Info_idxfirst(1,i)))*0.01;    
    GenerImp200Info_power(1,i) = trapz(x,GenerImp200(GenerImp200Info_idxfirst(1,i):GenerImp200Info_idxlast(1,i),i).^2);
end

% Generate the load time histories of 100 people
GerNum = 100; % The number of the crowd
AdaptIdx = zeros(length(Generpower200(:,1))-1,GerNum);
for i = 1:GerNum
    for j = 1:length(Generpower200(:,i))-1
        minIdxT1 = find(abs(Generdt_zero200(j+1,i)-GenerImp200Info_dt)<0.005);
        if ~isempty(minIdxT1)
            [~,minIdxT2] = min(abs(Generpower200(j,i)-GenerImp200Info_power(1,minIdxT1)));
            AdaptIdx(j,i) = minIdxT1(minIdxT2);
        else
            [~,minIdxT1] = find(abs(1/2.0-GenerImp200Info_dt)<0.005);
            [~,minIdxT2] = min(abs(Generpower200(j,i)-GenerImp200Info_power(1,minIdxT1)));
            AdaptIdx(j,i) = minIdxT1(minIdxT2);
        end
    end    
end

% Concatenate the generated impulse samples
GenerBounce = cell(1,GerNum);
GenerBounceTemp = cell(length(Generdt_zero200(:,1)),1);
for i = 1:GerNum
    for j = 1:length(Generdt_zero200(:,i))
        if j == 1
            tzero = 0:dt:Generdt_zero200(1,i);
            tzero = zeros(size(tzero))+1;
            GenerBounceTemp{j,1} = tzero';
            continue;
        end
        
            GenerBounceTemp{j,1} = GenerImp200(GenerImp200Info_idxfirst(1,AdaptIdx(j-1,i)):GenerImp200Info_idxlast(1,AdaptIdx(j-1,i)),AdaptIdx(j-1,i));
    end
    GenerBounce{1,i} = double(cat(1,GenerBounceTemp{:,1}));
end

len_min = zeros(1,length(GenerBounce));
for i = 1:length(len_min)
    len_min(1,i) = length(GenerBounce{1,i});
end
len_min = min(len_min);

GenerBounce_temp = zeros(len_min,GerNum);
for i = 1:GerNum
    Bounce_temp = GenerBounce{1,i};
    GenerBounce_temp(:,i) = Bounce_temp(1:len_min,1);
end
GenerBounce200 = GenerBounce_temp;