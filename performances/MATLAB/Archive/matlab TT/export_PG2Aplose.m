function [outputArg1,outputArg2] = export_PG2Aplose(PG_Annotation_formatted,dataset_name,label,annotator, WavFolderInfo,folder_data_wav)
%% Fonction qui permet de transformer les résultats de détection de PAMGuard
% dans les fichiers binaires en output APLOSE
timezone = '+00:00';

annotator_table = repmat(annotator,height(PG_Annotation_formatted),1);
dataset_table = repmat(dataset_name,height(PG_Annotation_formatted),1);
label_table = table2array(PG_Annotation_formatted(:,end));
start_time_table = table2array(PG_Annotation_formatted(:,1));
end_time_table = table2array(PG_Annotation_formatted(:,2));
start_frequency_table = table2array(PG_Annotation_formatted(:,3));
end_frequency_table = table2array(PG_Annotation_formatted(:,4));
start_datetime_table = datestr(table2array(PG_Annotation_formatted(:,5)),['yyyy-mm-ddTHH:MM:SS.FFF' timezone]);
end_datetime_table = datestr(table2array(PG_Annotation_formatted(:,6)),['yyyy-mm-ddTHH:MM:SS.FFF' timezone]);


idx_wav=[];
for i=1:height(PG_Annotation_formatted)
    idx_wav(i,1) = max(find(PG_Annotation_formatted.start_datetime(i) < WavFolderInfo.wavDates_formated ==0));
end

filename_formated = WavFolderInfo.wavDates_formated(idx_wav);



table_output =  table(dataset_table, filename_formated, start_time_table, end_time_table, start_frequency_table, end_frequency_table, label_table, annotator_table, start_datetime_table, end_datetime_table,...
    'VariableNames', ["dataset" "filename" "start_time" "end_time" "start_frequency" "end_frequency" "annotation" "annotator" "start_datetime" "end_datetime"]);



%%Generate APLOSE csv
writetable(table_output, strcat(folder_data_wav,"\","PG2Aplose table.csv"))



% file_name = [strcat(folder_data_wav,"\","PG2Aplose table.csv")]
% selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selction table + SRD at the end
    
% fprintf(selec_table,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', 'dataset','filename','start_time','end_time','start_frequency','end_frequency','annotation','annotator','start_datetime','end_datetime');
% fprintf(selec_table,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',C');
% fclose('all');

clc
fprintf('Done !\n');
end

