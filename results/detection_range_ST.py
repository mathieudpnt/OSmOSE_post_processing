import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import os
from post_processing_detections.utilities.def_func import extract_datetime, sorting_detections, t_rounder
import glob
import seaborn as sns
from scipy import stats
import json

#%% Load data
list_json = glob.glob(os.path.join('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', "**/metadata.json"), recursive=True)\
            + glob.glob(os.path.join('L:/acoustock2/Bioacoustique/APOCADO2', "**/metadata.json"), recursive=True)

data=[]
for i in range(len(list_json)):
    r_file = open(list_json[i], 'r')
    data.append(json.load(r_file))
    r_file.close()
data = pd.DataFrame.from_dict(data)

data['t_detections'] = [sorting_detections(i) for i in data['detection_file']]
data['df_detections'] = [i[0] for i in data['t_detections']]

#%%
# data1 = data[(data['n_instru']==2) & (data['net_length']!=400)]
data1 = data[data['n_instru']==2]
data_corr = pd.DataFrame()

for length in sorted(list(set(data1['net_length']))):

    sub_data = data1[data1['net_length']==length].sort_values(by=['deploy_ID'])
    list_sub_ID = sorted(list(set([i.split(' ')[0] for i in sub_data['deploy_ID']])))
    for ID in list_sub_ID:
        
        sub_data2 = sub_data[sub_data['deploy_ID'].str.contains(ID)].reset_index(drop=False)
        if len(sub_data2) == 2:
            
            df1_detections = sub_data2['df_detections'][0]
            df2_detections = sub_data2['df_detections'][1]
            
            tz_data = df1_detections['start_datetime'][0].tz
            
            wav_names1 = [os.path.basename(i) for i in sub_data2['wav_path'][0]]
            wav_names2 = [os.path.basename(i) for i in sub_data2['wav_path'][1]]
            
            durations1 = sub_data2['durations'][0]
            durations2 = sub_data2['durations'][1]
            
            time_bin = list(set(sub_data2['timebin']))[0]
            
            label_legend = [sub_data2['deploy_ID'][0], sub_data2['deploy_ID'][1]]
            
            # res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
            # res_min = 10 if (sum(sub_data2['duration_deployment'])/len(sub_data2))/3600 < 48 else 60
            res_min = 60
            
            delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min),\
                                        t_rounder(extract_datetime(min(wav_names1[0], wav_names2[0]), tz_data), res=600),\
                                        t_rounder(max(extract_datetime(wav_names1[-1], tz_data) + dt.timedelta(seconds=durations1[-1]), extract_datetime(wav_names2[-1], tz_data) + dt.timedelta(seconds=durations2[-1])), res=600)
                                        
            time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
            duration_h = int((time_vector[-1] - time_vector[0]).total_seconds()//3600)
                           
            n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice
            
            fig,ax = plt.subplots(nrows = 1, figsize=(20,9))
            data_hist = plt.hist([df1_detections['start_datetime'], df2_detections['start_datetime']], bins=time_vector, label=[label_legend[0], label_legend[1]]); #histo
            counts, bins = 100*(data_hist[0]/n_annot_max), data_hist[1]
                      
            df_corr = pd.DataFrame({'deployment': label_legend[0].split(' ')[0]+ ' '+label_legend[0].split(' ')[-1]+'/'+ label_legend[1].split(' ')[-1],\
                                         'ST1' : list(counts[0]),\
                                            'ST2' : list(counts[1]),\
                                                'net_len' : length})
            
            if len(data_corr) == 0: data_corr = df_corr
            else: data_corr = pd.concat([data_corr, df_corr])
            
            
            bars = range(0,110,10) #from 0 to 100 step 10
            y_pos = np.linspace(0,n_annot_max, num=len(bars))
            ax.set_yticks(y_pos, bars);
            ax.tick_params(axis='x', rotation= 60);
            ax.tick_params(labelsize=20)
            ax.set_ylabel("% de détections positives ("+str(res_min)+"min)", fontsize = 20)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M', tz=df1_detections['start_datetime'][0].tz))
            ax.tick_params(axis='x', rotation= 80)

            if time_vector[0].strftime('%d/%m/%Y') != time_vector[-1].strftime('%d/%m/%Y'):
                fig.suptitle(time_vector[0].strftime('%d/%m/%Y') + time_vector[-1].strftime(' - %d/%m/%Y UTC%z')+'\nduration : '+str(duration_h)+'h', fontsize = 24, y=1.06);
            else:
                fig.suptitle(time_vector[-1].strftime('%d/%m/%Y UTC%z')+'\nduration : '+str(duration_h)+'h', fontsize = 24, y=1.06);

                
            # grey background on odd/even days
            date_odd_even = [j for i,j in enumerate(time_vector) if j.day%2==0] #select odd or even days
            for i,j in enumerate(list(set([j.day for i,j in enumerate(date_odd_even)]))):
                vec = [l for k,l in enumerate(time_vector) if l.day==j]
                if (vec[-1].hour == 23) : vec[-1] += dt.timedelta(hours=1)
                elif vec[-1].day == time_vector[-1].day: vec[-1] = time_vector[-1]
                ax.fill_between([vec[0], vec[-1]],n_annot_max, color='grey', alpha=0.075) 
            
            if duration_h < 72: ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            elif duration_h > 72: ax.xaxis.set_major_locator(mdates.HourLocator(interval=5))
            plt.xlim(time_vector[0], time_vector[-1])
            plt.ylim(0, n_annot_max)
            ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
            ax.legend(fontsize = 30)                
                                        
print('Done')
#%% scatters

for select_len in sorted(set(list(data_corr['net_len']))):
    g = sns.lmplot(
        data=data_corr[data_corr['net_len'] == select_len],
        x="ST1",
        y="ST2",
        hue="deployment",
        height=6,
        scatter_kws={'s': 10},
        fit_reg=True,
        markers='.',
        line_kws={'lw': 1},
        palette='Set1',
        ci = 95
    )
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    def annotate(data, **kws):
        ID = list(data)
        r, p = stats.pearsonr(data[ID[0]], data[ID[1]])
        ax = plt.gca()
        ax.text(0,
                0,
                'R²={:.2f}'.format(r*r),
                transform=ax.transAxes
                )
        
    # g.map_dataframe(annotate)
    plt.title('net length : '+ str(list(set(data_corr[data_corr['net_len'] == select_len]['net_len']))[0]) +' m')
    plt.show()
    
sns.jointplot(data_corr[data_corr['net_len'] == select_len]['ST1'], data_corr[data_corr['net_len'] == select_len]['ST2'], kind="reg")
    
#%% ST correlation
sns.set_theme()
g = sns.lmplot(
    data=data_corr,
    x='ST1',
    y='ST2',
    hue = 'net_len',
    scatter_kws={'s': 1},
    line_kws={'lw': 1},
    ci = 95,
    legend = False,
    height = 8,
    markers='',
    palette = 'tab10'
)

color_leg = [g.ax.lines[i].get_color() for i in range(len(g.ax.lines))]
from scipy.stats import linregress
for i,n in enumerate(sorted(list(set(data_corr['net_len'])))):
    r, p = stats.pearsonr(data_corr[data_corr['net_len']==n]['ST1'], data_corr[data_corr['net_len']==n]['ST2'])
  
    slope, intercept, r_value, p_value, std_err = linregress(data_corr[data_corr['net_len'] == n]['ST1'], data_corr[data_corr['net_len'] == n]['ST2'])
  
    ax = plt.gca()
    ax.text(
            0.02,
            0.95-(i*0.05),
            '{}m - s={:.2f} - R²={:.3f}'.format(n, slope, r*r),
            transform=ax.transAxes,
            color = color_leg[i],
            size = 14,
            bbox=dict(facecolor='white', alpha=0.8)
            )
g.ax.set(
        xlabel='ST1 - hourly positive detection rate',
         ylabel='ST2 - hourly positive detection rate'
         )

plt.xlim(0, 100)
plt.ylim(0, 100)
# plt.title('Appaired SoundTraps correlation', y = 1, size = 15)

plt.show()
    
    
    
    

