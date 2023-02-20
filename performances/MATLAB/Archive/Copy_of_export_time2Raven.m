function [] = export_time2Raven(folder_data_wav, time_vector, time_bin, last_bin)

% duration_det : variable contenant les durees de chaque detection en secondes
if last_bin == 0
    duration_det = ones(length(time_vector)-1,1)*time_bin;
elseif last_bin > 0
    duration_det = [ones(length(time_vector)-2,1)*time_bin; last_bin];
end

wavList = dir(fullfile(folder_data_wav, '*.wav'));
wavNames = cell2mat(extractfield(wavList, 'name')');
splitDates = split(string(wavNames), '.',2);
wavDates = splitDates(:,2);
wavDates_formated = datetime(wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');




for i = 1:length(wavList)
    wavinfo(i) = audioinfo(strcat(folder_data_wav,"\",string(wavNames(i,:))));
end

% Début de chaque detection en s (ref 0s)
% Beg_sec = time_vector(1:end-1) - time_vector(1);
ref_wav = datenum(wavDates_formated(1))*3600*24  ; %datenum en s du du premier wav
% Beg_sec = (time_vector(1:end-1) - ref_wav)+3;
%Les fichiers ST ne durent pas exaxctement 7200s, cela implique un décalage
%temporel à chaque fichier à l'affichage de plusieurs wav sur Raven. Une
%correction est apportée pour pallier à cela.
for i = 1:length(wavDates_formated)-1
    dur_theo(i,1) = datenum(wavDates_formated(i+1) - wavDates_formated(i))*3600*24; %nbre de s théorique de chaque fichier wav à partir de leur nom
end
diff = dur_theo-extractfield(wavinfo(1:end-1),'Duration')' ;%offset ST : durée théorique - durée exacte
correction = cumsum(diff); %Cumul des offsets
for i =1:length(time_vector)-1
    test = min(find(time_vector(i) <= (datenum(wavDates_formated)*3600*24)~= 0))-1; %numéro du fichier wav dans lequel se trouve time_vector(i)
    if isempty(test)
        Beg_sec(i,1) = (time_vector(i) - ref_wav)     - correction(end); %On se trouve dans le dernier wav
    elseif test == 0 || test == 1
        Beg_sec(i,1) = (time_vector(i) - ref_wav); %pas de correction à apporter car on est dans le premier fichier wav
    else
        Beg_sec(i,1) = (time_vector(i) - ref_wav)     - correction(test);
    end
end
Beg_sec = abs(Beg_sec);
% Fin de chaque detection en s (ref 0s)
End_sec = Beg_sec + duration_det;


% Generate Raven selection Table with appropriate format
L = height(time_vector)-1;
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

% Frequency of each timebox
Low_freq = zeros(L,1);
High_freq = ones(L,1)*50000;

C = [Selection, View, Channel, Beg_sec, End_sec, Low_freq, High_freq]';

file_name = [strcat(folder_data_wav,'\', wavNames(1,1:end-4),' ', ' time_vector.txt')];
selec_table = fopen(file_name, 'wt');
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');
clc; disp("Time_vector table created");
end

