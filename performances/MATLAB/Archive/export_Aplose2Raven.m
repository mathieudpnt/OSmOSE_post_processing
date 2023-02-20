function [] = export_Aplose2Raven(WavFolderInfo, Ap_Annotation, folder_data_wav, string1)


% datenum_files : variable avec les dates des detections en MATLAB
datenum_start = datenum(Ap_Annotation.start_datetime);
datenum_end = datenum(Ap_Annotation.end_datetime);

% duration_det : variable contenant les durees de chaque detection en secondes
duration_det = Ap_Annotation.end_time - Ap_Annotation.start_time;



% % % % %Recalage timestamps Aplose
nb_sec_start = datenum_start*24*3600;
nb_sec_end = datenum_end*24*3600;
int_Ap = [nb_sec_start, nb_sec_end]; %Aplose annotation timestamps

% time_interval = [time_vector(1:end-1), time_vector(2:end) ];
% for i = 1:length(int_Ap)
% % for i = 1:length(int_Ap)-1
%     for j = 1:length(time_vector)-1
%         inter(j,1) = intersection_vect([nb_sec_start(i), nb_sec_end(i)], [time_vector(j), time_vector(j+1)]);
%     end    
%     idx = find(inter==1); %indexes of the timeboxes intersectif with Aplose box(i)
% 
%     if length(idx) > 1
%         ovlp_rate=[];
%         for j = 1:length(idx)
%             ovlp_rate(j) = overlap_rate( [time_vector(idx(j)), time_vector(idx(j)+1)], [nb_sec_start(i), nb_sec_end(i)] );
%         end
%         [X1,X2] = max(ovlp_rate);
%         int_Ap(i,:) = [time_vector(idx(X2)), time_vector(idx(X2)+1)];
%     end    
% end

% Début de chaque detection en s (ref 0s) 
% Beg_sec = int_Ap(:,1) - time_interval(1);
ref_wav = datenum(WavFolderInfo.wavDates_formated(1))*3600*24  ; %datenum en s du du premier wav
Beg_sec = (int_Ap(:,1) - ref_wav);





% Fin de chaque detection en s (ref 0s)
End_sec = Beg_sec + duration_det;

% Frequences limites de chaque detection
Low_freq = Ap_Annotation.start_frequency;
High_freq = Ap_Annotation.end_frequency;

% Generate Raven selection Table with appropriate format
L = height(Ap_Annotation);
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

C = [Selection, View, Channel, Beg_sec, End_sec, Low_freq, High_freq]';

if nargin < 4
    string1 = ' - APLOSE2Raven Selection Table.txt'
end
file_name = [strcat(folder_data_wav,'\', WavFolderInfo.txt_filename, string1)];
selec_table = fopen(file_name, 'wt');
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');
clc; disp("Aplose2Raven table created");
end

