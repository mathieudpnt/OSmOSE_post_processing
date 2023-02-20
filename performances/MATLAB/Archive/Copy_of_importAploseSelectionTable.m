function [output1] = importAploseSelectionTable(filename, WavFolderInfo, time_vector_datetime, index_exclude)

dataLines = [1, Inf];

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["dataset", "filename", "start_time", "end_time", "start_frequency", "end_frequency", "annotation", "annotator", "start_datetime", "end_datetime"];
opts.VariableTypes = ["string", "string", "double", "double", "double", "double", "string", "string", "string", "string"];
opts.SelectedVariableNames = ["filename", "start_time", "end_time", "start_frequency", "end_frequency", "start_datetime", "end_datetime","annotation", "annotator"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["annotation", "start_datetime", "end_datetime"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["filename", "start_time", "end_time", "start_frequency", "end_frequency","annotation",  "start_datetime", "end_datetime", "annotator"], "EmptyFieldRule", "auto");

% Import the data
output1 = readtable(filename, opts);
output1(1,:)=[];
output1.segment = char(output1.filename);
output1.filename = char(output1.filename);
output1.filename = strcat(output1.filename(:,1:end-11), output1.filename(:,end-3:end));
output1.filename = string(output1.filename);

fmax = max(output1.end_frequency);

for i = 1:length(output1.filename)
    idx_name(i,1) = find(output1.filename(i) == string({WavFolderInfo.wavList.name}') );
    output1.filename_formated(i) = WavFolderInfo.wavDates_formated(idx_name(i));
end

output1.start_datetime = strrep(output1.start_datetime,'T',' ');
output1.start_datetime = strrep(output1.start_datetime,'+00:00',' ');
output1.end_datetime = strrep(output1.end_datetime,'T',' ');
output1.end_datetime = strrep(output1.end_datetime,'+00:00',' ');

output1.start_datetime = datetime(output1.start_datetime, 'Format', 'yyyy MM dd - HH mm ss');
output1.end_datetime = datetime(output1.end_datetime, 'Format', 'yyyy MM dd - HH mm ss');


% output1.filename = datetime(output_temp3(:,2),'InputFormat', 'yyMMddHHmmss','Format', 'yyyy MM dd - HH mm ss');   %Mathieu 
% output1.filename = datetime(output_temp3,'InputFormat', 'yyyy-MM-dd-HH-mm-ss','Format', 'yyyy MM dd - HH mm ss');    %Maelle

%deletion of  box annotations (useless for now)
idx = find(output1.start_time ~= 0);
output1(idx,:)=[];
idx = find(output1.start_frequency ~= 0);
output1(idx,:)=[];
idx = find(output1.end_frequency ~= fmax);
output1(idx,:)=[];

output1 = sortrows(output1, 6); %Sort according to datetime begin

%Adjustment of the timestamps
for i =1:height(output1)
    idx = find(output1.filename_formated(i) == WavFolderInfo.wavDates_formated);
    if idx ~= 1

%         adjust = datetime(time_vector(index_exclude(idx-1)+1)/3600/24,'ConvertFrom','datenum')-WavFolderInfo.wavDates_formated(idx);
        adjust = time_vector_datetime(index_exclude(idx-1)+1)-WavFolderInfo.wavDates_formated(idx);
        %index_exclude : indexes of last bins before new wav, then
        %index_exclude(i)+1 : indexes of first timebin of a wav i+1

        output1.start_datetime(i) = output1.start_datetime(i) + adjust;
        output1.end_datetime(i) =   output1.end_datetime(i) +  adjust;
    end
end


end