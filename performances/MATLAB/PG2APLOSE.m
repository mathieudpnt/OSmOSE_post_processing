function [] = PG2APLOSE(info_deployment, wav_path, binary_path)
% This function import .pgdf PAMGUARD binary files info and export it as an Aplose csv & Raven selection table

start = datetime("now");

% binary folder
dt_binary_file = char(extractfield(dir(fullfile(binary_path, '/**/*.pgdf')), 'name')');
dt_binary_file = datetime(dt_binary_file(:, end-19:end-5), 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);

% info from wav files
wav_info.struct = dir(fullfile(wav_path, '/**/*.wav'));
wav_info.filename = string(extractfield(wav_info.struct, 'name')');
wav_info.folder = string(extractfield(wav_info.struct, 'folder')');
wav_info.datetime = convert_datetime(wav_info.filename, info_deployment.dt_format, info_deployment.timezone);

% user input to select the datetime of deployment and recovery
[~, prompt_title, ~] = fileparts(binary_path);

% loop until valid inputs are provided or user cancels
while true
    % prompt user for deployment date and time
    deployment1_str = inputdlg('Date & time of deployment (dd MM yyyy HH mm ss):', prompt_title);
    if isempty(deployment1_str)  % check if user cancels
        return;
    end
    
    % prompt user for recovery date and time
    deployment2_str = inputdlg('Date & time of recovery (dd MM yyyy HH mm ss):', prompt_title);
    if isempty(deployment2_str)  % check if user cancels
        return;
    end
    
    % convert input strings to datetime
    deployment1 = datetime(string(deployment1_str), 'InputFormat', 'dd MM yyyy HH mm ss', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);
    deployment2 = datetime(string(deployment2_str), 'InputFormat', 'dd MM yyyy HH mm ss', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);
    
    % validate inputs
    if isempty(deployment1) || isempty(deployment2)
        errordlg('Please enter valid date and time values', 'Invalid Input', 'modal');
    elseif deployment2 <= deployment1
        errordlg('Recovery date and time must be after deployment date and time', 'Invalid Input', 'modal');
    else
        break;
    end
end
dt_deploy = [deployment1, deployment2];

% file selection based on previous user inputs
idx_file{1} = find(wav_info.datetime < min(dt_binary_file), 1, 'last');
if isempty(idx_file{1})
    idx_file_beg = 1;
else
    [~, min_idx(1,1)]= min(abs(min(dt_binary_file) - wav_info.datetime(idx_file{1}:idx_file{1}+1)));
    if min_idx(1,1) == 1
        idx_file_beg = idx_file{1};
    elseif min_idx(1,1) == 2
        idx_file_beg = idx_file{1} + 1;
    end
end

idx_file{2} = find(wav_info.datetime > max(dt_binary_file), 1);
if isempty(idx_file{2})
    idx_file_end = length(wav_info.datetime);
else
    [~, min_idx(1,2)]= min(abs(min(dt_binary_file)-wav_info.datetime(idx_file{2}-1:idx_file{2})));
    if min_idx(1,2) == 1
        idx_file_end = idx_file{2}-1;
    elseif min_idx(1,2) == 2
        idx_file_end = idx_file{2};
    end
end
wav_info.struct([1:idx_file_beg-1,idx_file_end+1:end]) = [];
wav_info.filename([1:idx_file_beg-1,idx_file_end+1:end]) = [];
wav_info.folder([1:idx_file_beg-1,idx_file_end+1:end]) = [];
wav_info.datetime([1:idx_file_beg-1,idx_file_end+1:end]) = [];

for i=1:numel(wav_info.filename)
    wav_info.wavinfo(i,1) = audioinfo(fullfile(wav_info.folder(i,:), wav_info.filename(i,:)));
    clc
    disp(['reading wav files...', num2str(i),'/', num2str(numel(wav_info.filename))])
end

wav_info.txt_filename = extractBefore(wav_info.filename(1), '.wav');

% importation of data
[PG_Annotation_Raven,  PG_Annotation_Aplose] = importBinary(info_deployment, wav_info, binary_path, dt_deploy);

% name of file
if isequal([year(wav_info.datetime(1)), month(wav_info.datetime(1)), day(wav_info.datetime(1))],...
        [year(wav_info.datetime(end)), month(wav_info.datetime(end)), day(wav_info.datetime(end))])
    PG_title_datestr = string(datetime(wav_info.datetime(1), 'Format','yyMMdd'));
else
    PG_title_datestr = strcat(string(datetime(wav_info.datetime(1), 'Format','yyMMdd')), '_' , ...
        string(datetime(wav_info.datetime(end), 'Format','yyMMdd')));
end

% export data as APLOSE csv file
writetable(PG_Annotation_Aplose, strcat(binary_path, strcat('\PG_rawdata_', PG_title_datestr, '.csv')))

% export data as raven txt file
file_name = strcat(binary_path, '\PG_rawdata_', PG_title_datestr , '.txt');
selec_table = fopen(file_name, 'wt');
fprintf(selec_table, '%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table, '%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n', PG_Annotation_Raven);
fclose('all');

stop = datetime("now");
elapsed_time = seconds(stop - start);

clc
disp(['raw data files exported in: ', binary_path])
fprintf('\n')
disp(['elapsed time: ', num2str(uint64(elapsed_time)), ' s'])

end