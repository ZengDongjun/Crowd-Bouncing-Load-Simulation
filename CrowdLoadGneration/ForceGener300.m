load('Generdt_zero300.mat');
load('GenerImp300.mat');
load('Generpower300.mat');
dt = 0.01;
GenerImp300Info_idxfirst = zeros(1,size(GenerImp300,2));
GenerImp300Info_idxlast = zeros(1,size(GenerImp300,2));
GenerImp300Info_dt = zeros(1,size(GenerImp300,2));
GenerImp300Info_power = zeros(1,size(GenerImp300,2));

% Calculate the duration of each generated impulse
for i = 1:size(GenerImp300,2)
    % Identify the beginning point 
    GenerImp300Info_idxfirst(1,i) = find(GenerImp300(:,i)-1<0.1,1,'first');
    % Identify the ending point 
    [~,Info_peak_idxlast_temp] = findpeaks(GenerImp300(:,i),'SortStr','descend'); % Peak value identification
    Info_idxlast_temp = zeros(1,length(GenerImp300(:,i)));
    k = 0;
    for j = 1:length(GenerImp300(:,i))-1
        if GenerImp300(j,i)-1>0.1 && GenerImp300(j+1,i)-1<0.1
            k = k + 1;
            Info_idxlast_temp(k) = j;
        end
    end
    % If there is no intersection point
    if max(Info_idxlast_temp) <= Info_peak_idxlast_temp(1)
        continue
    end
    idxtemp = find(Info_idxlast_temp - Info_peak_idxlast_temp(1)>0,1,'first');
    GenerImp300Info_idxlast(1,i) = Info_idxlast_temp(idxtemp);
end

% Calculate the parameters of the generated sample
for i = 1:size(GenerImp300,2)
    GenerImp300Info_dt(1,i) = (GenerImp300Info_idxlast(1,i)- GenerImp300Info_idxfirst(1,i))*0.01;
end

for i = 1:size(GenerImp300,2)
    x = (0:(GenerImp300Info_idxlast(1,i)- GenerImp300Info_idxfirst(1,i)))*0.01;    
    GenerImp300Info_power(1,i) = trapz(x,GenerImp300(GenerImp300Info_idxfirst(1,i):GenerImp300Info_idxlast(1,i),i).^2);
end

% Generate the load time histories of 100 people
GerNum = 100; % The number of the crowd
AdaptIdx = zeros(length(Generpower300(:,1))-1,GerNum);
for i = 1:GerNum
    for j = 1:length(Generpower300(:,i))-1
        minIdxT1 = find(abs(Generdt_zero300(j+1,i)-GenerImp300Info_dt)<0.005);
        if ~isempty(minIdxT1)
            [~,minIdxT2] = min(abs(Generpower300(j,i)-GenerImp300Info_power(1,minIdxT1)));
            AdaptIdx(j,i) = minIdxT1(minIdxT2);
        else
            [~,minIdxT1] = find(abs(1/3.0-GenerImp300Info_dt)<0.005);
            [~,minIdxT2] = min(abs(Generpower300(j,i)-GenerImp300Info_power(1,minIdxT1)));
            AdaptIdx(j,i) = minIdxT1(minIdxT2);
        end
    end    
end

% Concatenate the generated impulse samples
GenerBounce = cell(1,GerNum);
GenerBounceTemp = cell(length(Generdt_zero300(:,1)),1);
for i = 1:GerNum
    for j = 1:length(Generdt_zero300(:,i))
        if j == 1
            tzero = 0:dt:Generdt_zero300(1,i);
            tzero = zeros(size(tzero))+1;
            GenerBounceTemp{j,1} = tzero';
            continue;
        end
        
            GenerBounceTemp{j,1} = GenerImp300(GenerImp300Info_idxfirst(1,AdaptIdx(j-1,i)):GenerImp300Info_idxlast(1,AdaptIdx(j-1,i)),AdaptIdx(j-1,i));
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
GenerBounce300 = GenerBounce_temp;