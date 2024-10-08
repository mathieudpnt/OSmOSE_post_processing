# %% Import raw data
import os

os.chdir(r"U:/Documents_U/Git/post_processing_detections")

from utils.glider_utils import load_glider_nav
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gpxpy
from pathlib import Path

# %%
input_dir = Path(r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_58_OHAGEODAMS\APRES_MISSION\NAV")

df = load_glider_nav(input_dir)
df.to_csv(Path(input_dir) / "out.csv", index=False)

# Define the deployment number based on file number ranges
# df['Deployment'] = 0
# df.loc[(df['yo'] >= 3) & (df['yo'] <= 6), 'Deployment'] = 1
# df.loc[(df['yo'] >= 7) & (df['yo'] <= 29), 'Deployment'] = 2
# df.loc[(df['yo'] >= 30) & (df['yo'] <= 44), 'Deployment'] = 3
# df.loc[(df['yo'] >= 46) & (df['yo'] <= 54), 'Deployment'] = 4



# %% Export deployements to GPX file

if "Deployment" in df.columns:
    for i in list(set(df["Deployment"])):
        gpx = gpxpy.gpx.GPX()

        # Group the DataFrame rows by deployment number
        grouped_df = df[df["Deployment"] == i].groupby("Deployment")

        # Iterate over each deployment group
        for deployment_number, group in grouped_df:
            track = gpxpy.gpx.GPXTrack()

            # Iterate over the rows within the deployment group
            segment = gpxpy.gpx.GPXTrackSegment()
            for index, row in group.iterrows():
                latitude = float(row["Lat DD"])
                longitude = float(row["Lon DD"])
                elevation = float(row["Depth"])
                timestamp = row["Datetime"]
                point = gpxpy.gpx.GPXTrackPoint(
                    latitude, longitude, elevation, time=timestamp
                )
                segment.points.append(point)

            # Add the segment to the track
            track.segments.append(segment)

            # Add the track to the GPX file
            gpx.tracks.append(track)

            # Waypoint
            Lat_wpt = df[df["Deployment"] == i]["Lat DD"].iloc[0]
            Lon_wpt = df[df["Deployment"] == i]["Lon DD"].iloc[0]
            waypoint = gpxpy.gpx.GPXWaypoint(
                latitude=Lat_wpt, longitude=Lon_wpt, name=f"deployment {i}"
            )

            gpx.waypoints.append(waypoint)

        # Save the GPX file
        with open(os.path.join(input_dir, f"deployment_{i}.gpx"), "w") as file:
            file.write(gpx.to_xml())
            file.close()
            print(f"deployment {i} exported")
else:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()

    # Iterate over the rows within the deployment group
    segment = gpxpy.gpx.GPXTrackSegment()

    for index, row in df.iterrows():
        latitude = float(row["Lat DD"])
        longitude = float(row["Lon DD"])
        elevation = float(row["Depth"])
        timestamp = row["Timestamp"]
        point = gpxpy.gpx.GPXTrackPoint(latitude, longitude, elevation, time=timestamp)
        segment.points.append(point)

    # Add the segment to the track
    track.segments.append(segment)

    # Add the track to the GPX file
    gpx.tracks.append(track)

    # Waypoint
    Lat_wpt = df["Lat DD"].iloc[0]
    Lon_wpt = df["Lon DD"].iloc[0]
    waypoint = gpxpy.gpx.GPXWaypoint(
        latitude=Lat_wpt, longitude=Lon_wpt, name="deployment_all"
    )

    gpx.waypoints.append(waypoint)

    # Save the GPX file
    with open(os.path.join(input_dir, "deployment_all.gpx"), "w") as file:
        file.write(gpx.to_xml())
        file.close()
        print(f"deployment_all exported")

# %% Plot a criterion for a deployment

d = 1
criterion = "Depth"

plt.plot(df[df["Deployment"] == d]["Datetime"], df[df["Deployment"] == d][criterion])
plt.title("Deployment {0} - {1}".format(d, criterion))
plt.grid(color="k", linestyle="-", linewidth=0.2, axis="y")

# Format the x-axis labels as '%H:%M'
date_formatter = mdates.DateFormatter("%d/%m %H:%M")
plt.gca().xaxis.set_major_formatter(date_formatter)

# Set the desired interval for the x-axis labels (e.g., every 2 hours)
# hours_interval = mdates.HourLocator(byhour=[0, 12])  # Adjust the interval value to change the interval
# plt.gca().xaxis.set_major_locator(hours_interval)

plt.show()

# import statistics as st
# st.mean(df[df['Deployment']==4]['Voltage'])

# %%

# utiliser les csv de detections avec les positions et les tracer sur QGIS
# plotter la profondeur du glider sur le deploiement avec les detection aussi dessus
