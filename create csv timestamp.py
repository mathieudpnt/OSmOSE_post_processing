import pandas as pd
import pytz
import easygui
import os
#%%
tz_data = 'Europe/Paris'

date_begin = pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime begin ? (dd MM yyyy HH mm ss) :"), format='%d %m %Y %H %M %S'))
date_end =   pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime end ? (dd MM yyyy HH mm ss) :"), format='%d %m %Y %H %M %S'))

#%%
# WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'
WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/wav'

# Create list of wav filenames
filenames=[]
for file in os.listdir(WavPath):
    if file.endswith('.wav'):
        filenames.append(file)
        
data = pd.DataFrame(filenames, columns=['filename'])
date_str = [j.split('.')[1] for i,j in enumerate(data['filename'])]
data['datetime'] = [pd.to_datetime(j+tz_data, format='%y%m%d%H%M%S%Z') for i,j in enumerate(date_str)]
data['timestamp'] = [j.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+'Z' for i,j in enumerate(data['datetime'])]


df2 = data[(data['datetime']>= date_begin) & (data['datetime']<= date_end)]

df2[['filename', 'timestamp']].to_csv(WavPath+'/timestamp_PG.csv', index=False, header=False)


