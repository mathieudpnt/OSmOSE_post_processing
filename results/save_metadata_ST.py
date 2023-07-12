import os
from post_processing_detections.utilities.def_func import read_header, sorting_annot_boxes, get_season, histo_detect
import glob
from pathlib import Path
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
import numpy as np
import statistics as stat
#%% import csv deployment

deploy = pd.read_excel('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsx', skiprows=[0])
deploy = deploy.loc[~((deploy['N° campagne'] == 4) & (deploy['N° déploiement'] == 9)), :] #deleting C4D9
deploy = deploy.loc[~((deploy['N° campagne'] == 7) & (deploy['N° déploiement'] == 1)), :]
deploy = deploy.reset_index(drop=True)

deploy['durations_deployments'] = [dt.datetime.combine(deploy['Date fin déploiement'][i], deploy['Heure fin déploiement'][i])\
                                    -dt.datetime.combine(deploy['Date début déploiement'][i], deploy['Heure début déploiement'][i]) for i in range(len(deploy))]

deploy['season'] = [get_season(i)[:-5] for i in deploy['Date début déploiement']]
deploy['season_year'] = [get_season(i)[-4:] for i in deploy['Date début déploiement']]

print('\n#### RESULTS DEPLOYMENTS ####')
print('-total duration: ', sum(deploy['durations_deployments'], dt.timedelta()))

print('\n# Duration per season #')
# [print('-{0}:'.format(season), int(sum(deploy[deploy['season']== season]['durations_deployments'], dt.timedelta()).total_seconds()/3600),'h') for season in ['spring', 'summer', 'autumn', 'winter']];
for y in sorted(list(set(deploy['season_year']))):    
    for s in list(dict.fromkeys(deploy[(deploy['season_year']==y)]['season'])):
        print('-{0}:'.format(s+' '+y), int(sum(deploy[(deploy['season_year'] == y) & (deploy['season'] == s)]['durations_deployments'], dt.timedelta()).total_seconds()/3600),'h')

print('\n# Duration per net #')
[print('-Filet {0}:'.format(filet), int(sum(deploy[deploy['Filet']== filet]['durations_deployments'], dt.timedelta()).total_seconds()/3600),'h') for filet in sorted(list(set(deploy['Filet'])))];

print('\n# Mean duration of a deployment per net type #')
[print('-Filet {0}:'.format(filet), round(np.mean(deploy[deploy['Filet']== filet]['durations_deployments']).total_seconds()/3600,1), 'h +/-', round(np.std(deploy[deploy['Filet']== filet]['durations_deployments']).total_seconds()/3600,1), 'h') for filet in sorted(list(set(deploy['Filet'])))];

print('\n# Duration per net length #')
[print('-{:.0f}'.format(L),'m :', int(sum(deploy[deploy['Longueur (m)']== L]['durations_deployments'], dt.timedelta()).total_seconds()/3600),'h') for L in sorted(list(set(deploy['Longueur (m)'])))];
#%%
x = [str(elem) for elem in sorted(list(set(deploy['Longueur (m)'])))]
y = [int(sum(deploy[deploy['Longueur (m)']== L]['durations_deployments'], dt.timedelta()).total_seconds()/3600) for L in sorted(list(set(deploy['Longueur (m)'])))]

fig,ax = plt.subplots(figsize=(16,6), facecolor='#36454F')
ax.bar(x,y); #histo
ax.set_facecolor('#36454F')
ax.tick_params(axis='both', colors='w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('', color='w')
ax.set_ylabel('Heures d\'enregistrement', fontsize = 16, color='w')
ax.set_xlabel('Longueur filière [m]', fontsize = 16, color='w')

#%% Save metadata

list_csv = glob.glob(os.path.join('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', '**/PG_formatteddata**.csv'), recursive=True)\
            +glob.glob(os.path.join('L:/acoustock2/Bioacoustique/APOCADO2', '**/PG_formatteddata**.csv'), recursive=True)

# for i in tqdm(list_csv):

#     t_detections = sorting_annot_boxes(i)
    
#     time_bin = t_detections[0]
#     fmax = t_detections[1]
#     [annotators] = t_detections[2]
#     [labels] = t_detections[3]
#     df_detections = t_detections[-1]
    
#     [ID_detections] = list(set(df_detections['dataset']))
#     rank = [i for i, ID in enumerate(deploy['ID']) if ID in ID_detections][0]
#     duration_deployment = int(deploy['durations_deployments'][rank].total_seconds())
#     dt_deployment_beg = dt.datetime.strftime(dt.datetime.combine(deploy['Date début déploiement'][rank], deploy['Heure début déploiement'][rank]), '%d/%m/%Y %H:%M:%S')
#     dt_deployment_end = dt.datetime.strftime(dt.datetime.combine(deploy['Date fin déploiement'][rank], deploy['Heure fin déploiement'][rank]), '%d/%m/%Y %H:%M:%S')
#     ID = deploy['ID'][rank]
#     net_len = int(deploy['Longueur (m)'][rank])
#     n_instru = deploy['nb ST/filet'][rank] if type(deploy['nb ST/filet'][rank]) is int else int(deploy['nb ST/filet'][rank][0])
    
#     wav_files = glob.glob(os.path.join(Path(i).parents[2], "**/*.wav"), recursive=True)
#     wav_names = [os.path.basename(file) for file in wav_files]
#     [wav_folder] = list(set([os.path.dirname(file) for file in wav_files]))
#     test_wav = [j in sorted(list(set([i.split('_')[0] for i in df_detections['filename']]))) for j in [i.split('.wav')[0] for i in wav_names]]
#     wav_names, wav_files = zip(*[(wav_names[i], wav_files[i]) for i in range(len(wav_names)) if test_wav[i]]) #only the wav files corresponding to the detections are kept
#     durations = [read_header(file)[-1] for file in wav_files]
    
    
    
#     metadata =  {'detection_file': i, 'wav_folder': wav_folder, 'deploy_ID' : ID, 'beg_deployment': dt_deployment_beg, 'end_deployment': dt_deployment_end, 'duration_deployment': duration_deployment, 'fmax': fmax, 'timebin': time_bin, 'annotators': annotators, 'net_length': net_len, 'n_instru': n_instru, 'labels': labels, 'wav_path': wav_files, 'durations': durations}
    
#     out_file = open(os.path.join(Path(i).parents[0], 'metadata.json'), 'w+')
#     json.dump(metadata, out_file, indent=4)
#     out_file.close()




#%% load all detection dataframes

list_json = glob.glob(os.path.join('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', "**/metadata.json"), recursive=True)\
            +glob.glob(os.path.join('L:/acoustock2/Bioacoustique/APOCADO2', '**/metadata.json'), recursive=True)               

data=[]
for i in range(len(list_json)):
    r_file = open(list_json[i], 'r')
    data.append(json.load(r_file))
    r_file.close()
data = pd.DataFrame.from_dict(data)
data['df_detections'] = [sorting_annot_boxes(i)[-1] for i in data['detection_file']]

for i in range(len(data)):
    tz_deploy = data['df_detections'][i]['start_datetime'][0].tz
    data['df_detections'][i]['start_deploy'] = tz_deploy.localize(dt.datetime.strptime(data['beg_deployment'][i], '%d/%m/%Y %H:%M:%S'))
    data['df_detections'][i]['end_deploy'] = tz_deploy.localize(dt.datetime.strptime(data['end_deployment'][i], '%d/%m/%Y %H:%M:%S'))

deploy['detection_num'] = [len(data.loc[data['deploy_ID']==ID, 'df_detections'].reset_index(drop=True)[0]) for ID in deploy['ID']]
deploy['deploy_num_win'] = [deploy['durations_deployments'][i].total_seconds()/10 for i in range(len(deploy))] #num of 10s win
deploy['detection_rate'] = [ ((len(data[data['deploy_ID']==ID].reset_index(drop=True)['df_detections'][0]) * data[data['deploy_ID']==ID].reset_index(drop=True)['timebin'][0])/deploy['durations_deployments'][i].total_seconds())*100 for i, ID in enumerate(deploy['ID'])]
deploy['dt_begin'] = [dt.datetime.strftime(dt.datetime.combine(deploy['Date début déploiement'][i], deploy['Heure début déploiement'][i]), '%d/%m/%Y %H:%M:%S') for i in range(len(deploy))]
deploy['dt_end'] = [dt.datetime.strftime(dt.datetime.combine(deploy['Date fin déploiement'][i], deploy['Heure fin déploiement'][i]), '%d/%m/%Y %H:%M:%S') for i in range(len(deploy))]
deploy['season'] = [get_season(deploy['Date début déploiement'][i]) for i in range(len(deploy))]

print('\n#### Hourly positive detection rate (whistles)####')
print('-Total: {:.1f}%'.format(np.mean(deploy['detection_rate'])))
[print('-{0}: {1}%'.format(season, round(np.mean(deploy[deploy['season']== season ]['detection_rate']),1)) ) for season in list(set(deploy['season']))];

#%% test pentes detection_rate
test = data['df_detections'][40]

data_histo= test['start_datetime']
tb= test['end_time'][0]
deploy_dt = [test['start_deploy'][0], test['end_deploy'][0]]


# bins,vec = histo_detect(data_histo, deploy_dt, res_min=60, time_bin=tb, plot=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
res_min = 10
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min),  deploy_dt[0], deploy_dt[1]
bins = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
bins2 = [bins[i:i + math.ceil(len(bins)/4)] for i in range(0, len(bins), math.ceil(len(bins)/4))]
n_annot_max = (res_min*60)/tb #max nb of annoted time_bin max per res_min slice
tz_data = deploy_dt[0].tz

fig,ax = plt.subplots(figsize=(20,9))
ax.hist(data_histo, bins, cumulative=False); #histo
ax.set_ylabel("Detection rate [%] ("+str(res_min)+"min)", fontsize = 20)
ax.axvline(x = bins2[-1][-1], color = 'r')
[ax.axvline(x = bins2[i][0], color = 'g') for i in range(len(bins2))]
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')








#%% export csv for QGIS
# deploy_out = deploy.drop(columns=['Date début campagne', 'Date fin campagne', 'Heure début campagne', 'Heure fin campagne', 'check heures', 'Conditions météo', 'Conditions météo.1', 'Présence cétacés', 'Présence cétacés.1', 'Commentaire', 'Unnamed: 25', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29'])
deploy_out = deploy.drop(columns=['lat D','lat DM','lat DD','long D','long DM','long DD','Date début campagne', 'Date fin campagne', 'Heure début campagne', 'Heure fin campagne', 'check heures', 'Conditions météo', 'Conditions météo.1', 'Présence cétacés', 'Présence cétacés.1', 'Commentaire'])

for i in range(len(deploy['nb ST/filet'])):
    if '1' in str(deploy['nb ST/filet'][i]): deploy_out['nb ST/filet'][i]=1
    if '2' in str(deploy['nb ST/filet'][i]): deploy_out['nb ST/filet'][i]=2
    
deploy_out.to_csv('L:/acoustock/Bioacoustique/DATASETS/APOCADO/Data QGIS/APOCADO - Suivi déploiements 13042023.csv', index=False, encoding='latin1')

#%% Nombre moyen de detection à chaque heure de la journée / saison

hour_list = ['{:02d}:00'.format(i) for i in range(24)]; hour_list.append('0:00')
seasons = list(set(deploy['season']))
effort_obs = [int(sum(deploy[deploy['season']==season]['durations_deployments'], dt.timedelta()).total_seconds()/3600) for season in seasons]

all_detections = pd.concat(df_detections).reset_index(drop=True)
all_detections['date'] = [dt.datetime.strftime(i.date(), '%d/%m/%Y') for i in all_detections['start_datetime']]
all_detections['season'] = [get_season(i) for i in all_detections['start_datetime']]

result = {}
for season in tqdm(seasons):
    detection_byseason = all_detections[all_detections['season'] == season] #sub-dataframe 1 : tri par saison
    
    detection_dates = sorted(list(set(detection_byseason['date']))) #liste des dates / saison

    for date in detection_dates:
        detection_bydate = detection_byseason[detection_byseason['date'] == date] #sub-dataframe 2 : tri par saison & par date
        list_datasets = sorted(list(set(detection_bydate['dataset']))) #liste des datesets / saison / date
        
        for dataset in list_datasets:
            df = detection_bydate[detection_bydate['dataset'] == dataset].set_index('start_datetime')
            test = [len(df.between_time(hour_list[j], hour_list[j+1], inclusive='left')) for j in (range(len(hour_list)-1))]
            
            deploy_beg, deploy_end = int(df['start_deploy'][0].timestamp()), int(df['end_deploy'][0].timestamp())
            
            list_present_h = [dt.datetime.fromtimestamp(i) for i in list(range(deploy_beg, deploy_end, 3600))]
            list_present_h2 = [dt.datetime.strftime(list_present_h[i], '%d/%m/%Y %H') for i in range(len(list_present_h))]

            list_deploy_d = sorted(list(set([dt.datetime.strftime(dt.datetime.fromtimestamp(i), '%d/%m/%Y') for i in list(range(deploy_beg, deploy_end, 3600))])))
            list_deploy_d2 = [d for i, d in enumerate(list_deploy_d) if d in date][0]
            
            list_present_h3 = []
            for item in list_present_h2:
                if item.startswith(list_deploy_d2):
                    list_present_h3.append(item)

            list_deploy = [df['date'][0] + ' ' + n for n in [f'{i:02}' for i in range(0,24)]]

            for i, h in enumerate(list_deploy):
                if h not in list_present_h3:
                    test[i] = np.nan
                    
            if season in result:
                result[season].append(test)
            else:
                result[season] = [test]
    
    data = result[season]
    df_data = pd.DataFrame()
    df_data = ((df_data.from_dict(data, orient = 'columns').T/360)*100).median(1, skipna=True)
    
    plt.bar(height=df_data, x=hour_list[:-1], width=1)
    plt.ylim(0,100)
    plt.xlim(hour_list[0],hour_list[-2])
    plt.xticks(rotation=60)    
    plt.title('Mean detection rate per hour of the day in '+ season)
    plt.show()

#%%
data = pd.DataFrame([100*(np.nanmedian(result[season], axis=0)/360) for season in result.keys()], index=list(result.keys()) ).T.set_index([hour_list[:-1]])
data.plot(kind='bar', grid=0)
plt.style.use('default')
plt.ylim(0,40)
plt.xticks(rotation=90)   
plt.ylabel('positive detection rate, %', fontsize = 10)
plt.title('Delphinids acoustics presence', fontsize = 12, y=1);

plt.show()


