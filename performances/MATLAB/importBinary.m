function [data_table_Raven, data_table_Aplose] = importBinary(info_deploy, wav_info, binary_path, dt_deploy)
% this function extracts and converts detections from PAMGuard binary files

% sampling frequency
Fs = wav_info.wavinfo(1).SampleRate;

% pgdf info
binary_info.struct = dir(fullfile(binary_path, '/**/*.pgdf'));
binary_info.filename = extractfield(binary_info.struct, 'name')';
binary_info.datetime = datetime(cell2mat(cellfun(@(x) extractBefore(extractAfter(x, numel(x) - 20), '.pgdf'), ...
                        binary_info.filename, 'UniformOutput', false)), ...
                        'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ');
binary_info.datetime.TimeZone = info_deploy.timezone;

% load data
data = loadPamguardBinaryFolder(binary_path, '*.pgdf', 1);
detection_begin = datetime([data.date]', 'ConvertFrom', 'datenum');
detection_begin.TimeZone = info_deploy.timezone;
detection_duration = cell2mat({data(1:end).sampleDuration})' / Fs;
detection_end = detection_begin + seconds(detection_duration);
detection_begin_str = strcat(datestr(detection_begin, 'yyyy-mm-ddTHH:MM:SS.FFF'), info_deploy.timezone);
detection_end_str = strcat(datestr(detection_end, 'yyyy-mm-ddTHH:MM:SS.FFF'), info_deploy.timezone);
Beg_sec = seconds(detection_begin - min(binary_info.datetime)); % number of seconds between the first datetime and beginning of each detection
End_sec = Beg_sec + detection_duration; % number of seconds between the first datetime and end of each detection

% delete detections that are not within deployment and recovery datetimes
idx_to_delete = detection_begin < dt_deploy(1) | detection_begin > dt_deploy(2);
data(idx_to_delete) = [];
detection_begin(idx_to_delete) = [];
detection_end(idx_to_delete) = [];
detection_begin_str(idx_to_delete, :) = [];
detection_end_str(idx_to_delete, :) = [];
detection_duration(idx_to_delete) = [];
Beg_sec(idx_to_delete) = [];
End_sec(idx_to_delete) = [];

% min and max frequencies of each detection
freqs = cell2mat({data(1:end).freqLimits}'); 
freq_low = freqs(:,1);
freq_high = freqs(:,2);

% index of the corresponding wav file for each detection
idx_wav = zeros(numel(detection_begin), 1);
for i = 1:numel(detection_begin)
    idx_wav(i) = find(detection_begin(i) >= wav_info.datetime, 1, 'last');
end

% adjustment of the timestamps
wav_duration = extractfield(wav_info.wavinfo, 'Duration')';

% time differences between consecutive datetimes and add wav_duration
time_diff = diff(wav_info.datetime);
adjust = [0; seconds(-time_diff) + wav_duration(1:end-1)];
cumsum_adjust = cumsum(adjust);

% adjusted datetime to match Raven functionning
detection_begin_adjusted = detection_begin + seconds(cumsum_adjust(idx_wav));

% number of seconds from start of files list to beginning of each detection
Beg_sec_adjusted = seconds(detection_begin_adjusted - min(binary_info.datetime));

% number of seconds from start of files list to end of each detection
End_sec_adjusted = Beg_sec_adjusted + detection_duration;

% generate Raven selection table
L = length(data);
Selection = (1:L)';
View = ones(L,1);
Channel = ones(L,1);
data_table_Raven = [Selection, View, Channel, Beg_sec_adjusted, End_sec_adjusted, freq_low, freq_high]';
data_table_Raven = sortrows(data_table_Raven', 4)';

% generate APLOSE table
annotator = repmat(info_deploy.annotator, [L,1]);
annotation = repmat(info_deploy.annotation, [L,1]);
dataset = repmat(info_deploy.dataset, [L,1]);
filename = wav_info.filename(idx_wav);

data_table_Aplose = [ array2table([dataset, filename, Beg_sec, End_sec, freq_low, freq_high, annotation, annotator],...
    'VariableNames',{'dataset', 'filename', 'start_time','end_time','start_frequency','end_frequency', 'annotation', 'annotator'}),...
     table(detection_begin_str, detection_end_str, 'VariableNames',{'start_datetime','end_datetime'})];
data_table_Aplose = sortrows(data_table_Aplose, 'start_datetime');

clc
fprintf('Importation of %s detections done\n', num2str(length(data)))