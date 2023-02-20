function [] = export_time2Raven(folder_data_wav, WavFolderInfo, time_vector_datetime, time_bin, duration_det)
%export_time2Raven(folder_result, WavFolderInfo, time_vector, time_bin, duration_time)
Beg_sec = seconds(time_vector_datetime(1:end-1) - WavFolderInfo.wavDates_formated(1) );
%time_vector(1:end-1) car le dernier timestamps est le dernier endtime, ce n'est pas un beg_time


% Fin de chaque detection en s (ref 0s)
End_sec = Beg_sec + duration_det(1:end);


% Generate Raven selection Table with appropriate format
L = length(time_vector_datetime)-1;
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

% Frequency of each timebox
Low_freq = zeros(L,1);
High_freq = ones(L,1)*50000;

C = [Selection, View, Channel, Beg_sec, End_sec, Low_freq, High_freq]';

file_name = [strcat(folder_data_wav,'\', WavFolderInfo.txt_filename,' - Time_vector.txt')];
selec_table = fopen(file_name, 'wt');
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');
clc; disp("Time_vector table created");
end

