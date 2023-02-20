function [outputArg1,outputArg2] = export_PG2Raven(data, folder_data_wav, WavFolderInfo)


% Generate Raven selection Table with appropriate format
L = height(data);
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

C = [Selection, View, Channel, data.Begin_time, data.End_time, data.Low_Freq, data.High_Freq]';


%Print Result to txt file
% data_name = strcat(WavFolderInfo.wavNames(1), ' ', string1);
file_name = [strcat(folder_data_wav,'\',WavFolderInfo.txt_filename ,' - PamGuard2Raven Selection Table.txt')];
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selection table + SRD at the end
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');

clc; disp("PG2Raven table created");


end

