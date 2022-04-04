from src.data.weather.weather_dwd_importer import WeatherDataColumns

WEATHER_LABEL_MAP: dict[WeatherDataColumns, str] = {
    WeatherDataColumns.GH_W_PER_M2: r"Globalstrahlung",
    WeatherDataColumns.DH_W_PER_M2: r"Diffusstrahlung",
    WeatherDataColumns.WIND_DIR_DEGREE: r"Windrichtung",
    WeatherDataColumns.WIND_DIR_DEGREE_DELTA: r"Windrichtungsänderung",
    WeatherDataColumns.WIND_V_M_PER_S: r"Windgeschwindigkeit",
    WeatherDataColumns.T_AIR_DEGREE_CELSIUS: r"Temperatur",
}

WEATHER_UNIT_LATEX_MAP: dict[WeatherDataColumns, str] = {
    WeatherDataColumns.GH_W_PER_M2: r"$\frac{W}{m^{2}}$",
    WeatherDataColumns.DH_W_PER_M2: r"$\frac{W}{m^{2}}$",
    WeatherDataColumns.WIND_DIR_DEGREE: r"$^{\circ}$",
    WeatherDataColumns.WIND_DIR_DEGREE_DELTA: r"$^{\circ}$",
    WeatherDataColumns.WIND_V_M_PER_S: r"$\frac{m}{s}$",
    WeatherDataColumns.T_AIR_DEGREE_CELSIUS: r"$^{\circ}C$",
}


def get_weather_data_latex_label(col: WeatherDataColumns):
    return WEATHER_LABEL_MAP[col] + " " + WEATHER_UNIT_LATEX_MAP[col]
