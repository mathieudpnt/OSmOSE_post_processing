import folium
import webbrowser
import pandas as pd
import matplotlib

# Chargement des données
data = pd.read_excel(
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\APOCADO - Suivi déploiements.xlsm",
    skiprows=[0],
)
data = data[(data["campaign"] != 1)]  # deleting 1st campaign
data = data[(data["latitude"].notnull())]  # deleting

data = data.reset_index(drop=True)
lon = list(data["longitude"])
lat = list(data["latitude"])
site = list(data["ID platform"])

campagne = list(data["campaign"].unique())
colors = matplotlib.cm.Dark2(range(len(campagne)))
color_map = dict(zip(campagne, colors))

sw = data[["latitude", "longitude"]].min().values.tolist()
ne = data[["latitude", "longitude"]].max().values.tolist()

m = folium.Map(zoom_start=0, location=(48, -4.5))
m.fit_bounds([sw, ne])

# Fond de carte utilisé
folium.TileLayer(
    # tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    # tiles='openstreetmap',
    # tiles = 'http://localhost:8000/tiles/{z}/{y}/{x}.png',
    tiles="Stamen Terrain",
    # tiles='Stamen Toner',
    # tiles='CartoDB Positron',
    # tiles='CartoDB Dark_Matter',
    attr="APOCADO",
    overlay=True,
    control=True,
    dragging=False,
    zoomControl=False,
).add_to(m)

# Ajout d'un marqueur pour chaque site avec le nom au dessus + une fenêtre pop-up qui s'ouvre avec les infos
for i in range(len(site)):
    # Get the color based on 'Campagne' value
    campagne_value = data.loc[i, "campaign"]
    marker_color = color_map[campagne_value]

    icon = folium.CircleMarker(
        location=(lat[i], lon[i]),
        radius=3,
        color=matplotlib.colors.rgb2hex(marker_color),
        fill=True,
        fill_color=matplotlib.colors.rgb2hex(marker_color),
        fill_opacity=1,
        tooltip=site[i],
        draggable=False,
    ).add_to(m)

    popup_content = f"<b>Site:</b> {site[i]}<br><b>Latitude:</b> {lat[i]}<br><b>Longitude:</b> {lon[i]}"
    popup = folium.Popup(popup_content, max_width=250)

    marker = folium.Marker(
        location=(lat[i], lon[i]),
        icon=folium.DivIcon(
            icon_size=(1, 1),
            # html=f"""<div style="position:relative; font-family: Arial; color: black; font-weight: bold; margin-top: -10px; margin-right: 10px;">{site[i]}</div>"""
        ),
        popup=popup,
    )
    marker.add_to(m)
# Enregistrement de la carte dans un fichier HTML
m.save("map.html")

# Ouvrir la carte dans le navigateur
webbrowser.open("map.html")
