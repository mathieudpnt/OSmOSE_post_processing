#Add csv extension to all files after decompression

from tkinter import filedialog
from tkinter import Tk
import os

root = Tk()
root.withdraw()
directory = filedialog.askdirectory(title="Select master folder")

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        continue  # Skip files that already have the .csv extension
    new_filename = os.path.join(directory, filename + '.csv')
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    os.rename(old_filepath, new_filepath)
    
#%% concatenate csv NAV data into a single GPX file
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import os
from datetime import datetime as dt
import gpxpy

root = Tk()
root.withdraw()
directory = filedialog.askdirectory(title="Select master folder")

# Initialize an empty list to store the contents of all CSV files
all_rows = []
yo = []  # List to store the file numbers
file = []

first_file = True
file_number = 1  # Initialize the file number

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        # Skip the output_file if it already exists in the list

        with open(file_path, 'r') as csv_file:
            delimiter = ';'  # Specify the desired delimiter
            csv_reader = pd.read_csv(csv_file, delimiter=delimiter)
            # If it's the first file, append the header row
            if first_file:
                all_rows.append(csv_reader.columns.tolist())
                first_file = False
            # Add the rows from the current CSV file to the all_rows list
            all_rows.extend(csv_reader.values.tolist())
            # Add yo number to the yo list
            yo.extend([filename.split('.')[-2]] * len(csv_reader))
            file.extend([filename] * len(csv_reader))
            file_number += 1  # Increment the file number for the next file

# Create a DataFrame from the combined data
df = pd.DataFrame(all_rows)
df.columns = df.iloc[0]  # set 1st row as headers
df = df.iloc[1:, 0:-1]  # delete last column and 1st row

# Add the yo number to the DataFrame
df['yo'] = [int(x) for x in yo]
df['file'] = file

# Define the deployment number based on file number ranges
df['Deployment'] = 0
df.loc[(df['yo'] >= 3) & (df['yo'] <= 6), 'Deployment'] = 1
df.loc[(df['yo'] >= 7) & (df['yo'] <= 29), 'Deployment'] = 2
df.loc[(df['yo'] >= 30) & (df['yo'] <= 44), 'Deployment'] = 3
df.loc[(df['yo'] >= 46) & (df['yo'] <= 54), 'Deployment'] = 4

df = df.sort_values(by=['Timestamp']).reset_index(drop=True)
df = df.drop(df[(df['Lat'] == 0) & (df['Lon'] == 0)].index)

df['Lat DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lat']/100]
df['Lon DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lon']/100]
# df['Timestamp_str'] = [dt.strptime(x, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%dT%H:%M:%SZ') for x in df['Timestamp']]
df['Datetime'] = [dt.strptime(x, '%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]

df = df[df['Deployment']!=0]

#if new GPX file
# gpx = gpxpy.gpx.GPX() 
#if the data is added to an existing GPX file
with open('C:/Users/dupontma2/Downloads/boat.gpx', 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

# Group the DataFrame rows by deployment number
grouped_df = df.groupby('Deployment')

# Iterate over each deployment group
for deployment_number, group in grouped_df:
    track = gpxpy.gpx.GPXTrack()
    
    # Iterate over the rows within the deployment group
    segment = gpxpy.gpx.GPXTrackSegment()
    for index, row in group.iterrows():
        latitude = float(row['Lat DD'])
        longitude = float(row['Lon DD'])
        timestamp = row['Datetime']
        point = gpxpy.gpx.GPXTrackPoint(latitude, longitude, time=timestamp)
        segment.points.append(point)
    
    # Add the segment to the track
    track.segments.append(segment)
    
    # Add the track to the GPX file
    gpx.tracks.append(track)
    
    #Waypoint
    for i in list(set(df['Deployment'])):
        Lat_wpt = df[df['Deployment']==i]['Lat DD'].iloc[0]
        Lon_wpt = df[df['Deployment']==i]['Lon DD'].iloc[0]
        waypoint = gpxpy.gpx.GPXWaypoint(Lat_wpt, Lon_wpt, name='Deployment {0}'.format(i))    
        
        gpx.waypoints.append(waypoint)
    

# Save the GPX file
with open('C:/Users/dupontma2/Downloads/output.gpx', 'w') as f:
    f.write(gpx.to_xml())
    
#%% concatenate csv NAV data into a one GPX file per glider deployment

from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import os
from datetime import datetime as dt
import gpxpy

root = Tk()
root.withdraw()
directory = filedialog.askdirectory(title="Select master folder")


# Initialize an empty list to store the contents of all CSV files
all_rows = []
yo = []  # List to store the file numbers
file = []

first_file = True
file_number = 1  # Initialize the file number

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r') as csv_file:
            delimiter = ';'  # Specify the desired delimiter
            csv_reader = pd.read_csv(csv_file, delimiter=delimiter)
            # If it's the first file, append the header row
            if first_file:
                all_rows.append(csv_reader.columns.tolist())
                first_file = False
            # Add the rows from the current CSV file to the all_rows list
            all_rows.extend(csv_reader.values.tolist())
            # Add yo number to the yo list
            yo.extend([filename.split('.')[-2]] * len(csv_reader))
            file.extend([filename] * len(csv_reader))
            file_number += 1  # Increment the file number for the next file

# Create a DataFrame from the combined data
df = pd.DataFrame(all_rows)
df.columns = df.iloc[0]  # set 1st row as headers
df = df.iloc[1:, 0:-1]  # delete last column and 1st row

# Add the yo number to the DataFrame
df['yo'] = [int(x) for x in yo]
df['file'] = file

# Define the deployment number based on file number ranges
df['Deployment'] = 0
df.loc[(df['yo'] >= 3) & (df['yo'] <= 6), 'Deployment'] = 1
df.loc[(df['yo'] >= 7) & (df['yo'] <= 29), 'Deployment'] = 2
df.loc[(df['yo'] >= 30) & (df['yo'] <= 44), 'Deployment'] = 3
df.loc[(df['yo'] >= 46) & (df['yo'] <= 54), 'Deployment'] = 4

df = df.sort_values(by=['Timestamp']).reset_index(drop=True)
df = df.drop(df[(df['Lat'] == 0) & (df['Lon'] == 0)].index)

df['Lat DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lat']/100]
df['Lon DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lon']/100]
df['Datetime'] = [dt.strptime(x, '%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]

df = df[df['Deployment']!=0]

for i in list(set(df['Deployment'])):
    gpx = gpxpy.gpx.GPX()

    # Group the DataFrame rows by deployment number
    grouped_df = df[df['Deployment']==i].groupby('Deployment')

    # Iterate over each deployment group
    for deployment_number, group in grouped_df:
        track = gpxpy.gpx.GPXTrack()
        
        # Iterate over the rows within the deployment group
        segment = gpxpy.gpx.GPXTrackSegment()
        for index, row in group.iterrows():
            latitude = float(row['Lat DD'])
            longitude = float(row['Lon DD'])
            timestamp = row['Datetime']
            point = gpxpy.gpx.GPXTrackPoint(latitude, longitude, time=timestamp)
            segment.points.append(point)
        
        # Add the segment to the track
        track.segments.append(segment)
        
        # Add the track to the GPX file
        gpx.tracks.append(track)
        
        #Waypoint
        Lat_wpt = df[df['Deployment']==i]['Lat DD'].iloc[0]
        Lon_wpt = df[df['Deployment']==i]['Lon DD'].iloc[0]
        waypoint = gpxpy.gpx.GPXWaypoint(Lat_wpt, Lon_wpt, name='Deployment {0}'.format(i))    
        
        gpx.waypoints.append(waypoint)
        

    # Save the GPX file
    with open('C:/Users/dupontma2/Downloads/output_glider{0}.gpx'.format(i), 'w') as f:
        f.write(gpx.to_xml())
    

#%%

from post_processing_detections.utilities.def_func import load_glider_nav
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = load_glider_nav()

# Define the deployment number based on file number ranges
df['Deployment'] = 0
df.loc[(df['yo'] >= 3) & (df['yo'] <= 6), 'Deployment'] = 1
df.loc[(df['yo'] >= 7) & (df['yo'] <= 29), 'Deployment'] = 2
df.loc[(df['yo'] >= 30) & (df['yo'] <= 44), 'Deployment'] = 3
df.loc[(df['yo'] >= 46) & (df['yo'] <= 54), 'Deployment'] = 4

#%%
i = 3
criterion = 'Depth'
plt.plot(df[df['Deployment']==i]['Datetime'], df[df['Deployment']==i][criterion])
# plt.ylabel('Depth (m)')
plt.title('Deployment {0} - {1}'.format(i, criterion))
plt.grid(color='k', linestyle='-', linewidth=0.2, axis='y')

# Format the x-axis labels as '%H:%M'
date_formatter = mdates.DateFormatter('%d/%m %H:%M')
plt.gca().xaxis.set_major_formatter(date_formatter)

# Set the desired interval for the x-axis labels (e.g., every 2 hours)
hours_interval = mdates.HourLocator(byhour=[0, 12])  # Adjust the interval value to change the interval
plt.gca().xaxis.set_major_locator(hours_interval)

plt.show()

# import statistics as st
# st.mean(df[df['Deployment']==4]['Voltage'])





