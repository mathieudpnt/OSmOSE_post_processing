from pathlib import Path

import yaml

config_file = Path(r"C:\Users\fouinel\PycharmProjects\OSmOSE_post_processing\user_case\config.yaml")

config = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}

site_colors = config.get("site_colors", {"Site A Haute": "#F0E442", "Site B Heugh": "#0072B2", "Site C Chat": "#D55E00", "Site D Simone": "#BDD4EC", "CA4": "#80D8C3", "Walde": "#4DA8DA", "Point C": "#932F67", "Point D": "#D92C54", "Point E": "#DDDEAB", "Point F": "#8ABB6C", "Point G": "#456882"})

season_color = config.get("season_color", {"spring": "green", "summer": "orange", "autumn": "brown", "winter": "blue"})