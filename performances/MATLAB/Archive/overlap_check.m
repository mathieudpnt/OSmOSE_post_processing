function [Ap_Annotation] = overlap_check(Ap_Annotation)

% Overlap Check
last_idx=[];
name_unique=[];
overlap=[];
idx_overlap=[];
IND1=[];
IND2=[];

name_unique = unique(Ap_Annotation.filename_formated);
for i =1:length(name_unique)
    last_idx(i,1) = max(find(Ap_Annotation.filename_formated==name_unique(i)));
end

IND1 = last_idx(1:end-1);
IND2 = last_idx(1:end-1)+1;
% [IND1, IND2];

for i =1:length(IND1)
    overlap(i,1) = overlap_rate([Ap_Annotation.start_datetime(IND1(i)) Ap_Annotation.end_datetime(IND1(i))], [Ap_Annotation.start_datetime(IND2(i)) Ap_Annotation.end_datetime(IND2(i))]);
end

if sum(overlap)>0
    
    idx_overlap = find(overlap>0);

    [last_idx(idx_overlap), last_idx(idx_overlap)+1]; %intruders
    clc
    disp([num2str(length(last_idx(idx_overlap))) ' overlaps in APLOSE csv result file'])
    disp([num2str([last_idx(idx_overlap), last_idx(idx_overlap)+1])  ])

    Ap_Annotation.end_datetime(last_idx(idx_overlap)) = Ap_Annotation.start_datetime(last_idx(idx_overlap)+1);
    Ap_Annotation.end_time(last_idx(idx_overlap)) = round(seconds(Ap_Annotation.end_datetime(last_idx(idx_overlap))- Ap_Annotation.start_datetime(last_idx(idx_overlap))));
elseif sum(overlap)==0
    clc
    disp('No overlap')
end

end

