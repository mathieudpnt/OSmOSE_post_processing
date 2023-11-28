# %% Import raw data

from utilities.def_func import load_glider_nav
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gpxpy
import os

df = load_glider_nav()

# Define the deployment number based on file number ranges
df['Deployment'] = 0
df.loc[(df['yo'] >= 3) & (df['yo'] <= 6), 'Deployment'] = 1
df.loc[(df['yo'] >= 7) & (df['yo'] <= 29), 'Deployment'] = 2
df.loc[(df['yo'] >= 30) & (df['yo'] <= 44), 'Deployment'] = 3
df.loc[(df['yo'] >= 46) & (df['yo'] <= 54), 'Deployment'] = 4

# %% Export deployements to GPX file

output_path = 'L:/acoustock/Bioacoustique/DATASETS/GLIDER/GLIDER SEA034/MISSION_46_DELGOST/APRES_MISSION/NAV/gli'

for i in list(set(df['Deployment'])):
    gpx = gpxpy.gpx.GPX()

    # Group the DataFrame rows by deployment number
    grouped_df = df[df['Deployment'] == i].groupby('Deployment')

    # Iterate over each deployment group
    for deployment_number, group in grouped_df:
        track = gpxpy.gpx.GPXTrack()

        # Iterate over the rows within the deployment group
        segment = gpxpy.gpx.GPXTrackSegment()
        for index, row in group.iterrows():
            latitude = float(row['Lat DD'])
            longitude = float(row['Lon DD'])
            elevation = float(row['Depth'])
            timestamp = row['Datetime']
            point = gpxpy.gpx.GPXTrackPoint(latitude, longitude, elevation, time=timestamp)
            segment.points.append(point)

        # Add the segment to the track
        track.segments.append(segment)

        # Add the track to the GPX file
        gpx.tracks.append(track)

        # Waypoint
        Lat_wpt = df[df['Deployment'] == i]['Lat DD'].iloc[0]
        Lon_wpt = df[df['Deployment'] == i]['Lon DD'].iloc[0]
        waypoint = gpxpy.gpx.GPXWaypoint(latitude=Lat_wpt, longitude=Lon_wpt, name='Deployment {0}'.format(i))

        gpx.waypoints.append(waypoint)

    # Save the GPX file
    with open(os.path.join(output_path, 'deployment_{0}.gpx').format(i), 'w') as file:
        file.write(gpx.to_xml())
        file.close()
        print(f'deployment {i} exported')

print(f'All files exported to {output_path}')


# %% Plot one deployment
i = 1
criterion = 'Depth'
plt.plot(df[df['Deployment'] == i]['Datetime'], df[df['Deployment'] == i][criterion])
# plt.ylabel('Depth (m)')
plt.title('Deployment {0} - {1}'.format(i, criterion))
plt.grid(color='k', linestyle='-', linewidth=0.2, axis='y')

# Format the x-axis labels as '%H:%M'
date_formatter = mdates.DateFormatter('%d/%m %H:%M')
plt.gca().xaxis.set_major_formatter(date_formatter)

# Set the desired interval for the x-axis labels (e.g., every 2 hours)
# hours_interval = mdates.HourLocator(byhour=[0, 12])  # Adjust the interval value to change the interval
# plt.gca().xaxis.set_major_locator(hours_interval)

plt.show()

# import statistics as st
# st.mean(df[df['Deployment']==4]['Voltage'])
