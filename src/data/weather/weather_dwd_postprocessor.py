from dataclasses import dataclass, field
from typing import Callable, Optional

from pandas import DataFrame, Series
from src.data.processor.pandas import PandasProcessor
from src.data.weather.weather_dwd_importer import ExtraCalculatedMapping, WeatherDataColumns
from src.data.weather.weather_filters import (
    create_night_time_replace_handler,
    wind_dir_cleansing,
)


@dataclass
class ColumnProcessorOptions:
    extra_mappings: list[ExtraCalculatedMapping] = field(default_factory=list)
    post_processor: Optional[Callable[[DataFrame], DataFrame]] = None


def solar_irradiance_postprocessor(data: Series, df: DataFrame) -> Series:
    min_solar_irradiance = 0
    max_solar_irradiance = 1200
    night_time_replacer = create_night_time_replace_handler(night_time_element_value=0)
    return night_time_replacer(PandasProcessor(data).clip(min_solar_irradiance, max_solar_irradiance).run())


def diffuse_solar_irradiance_postprocessor(data: Series, df: DataFrame) -> Series:
    print(f"{data.name=}")
    df[data.name].values[:] = df[[data.name, WeatherDataColumns.GH_W_PER_M2]].min(axis=1).values
    # data.replace(data, df[[data.name, WeatherDataColumns.GH_W_PER_M2]].min(axis=1))
    # data[data.name].values = df[[data.name, WeatherDataColumns.GH_W_PER_M2]].min(axis=1).values
    return solar_irradiance_postprocessor(data, df)


def wind_dir_postprocessing(data: Series, df: DataFrame) -> Series:
    return wind_dir_cleansing(data)


def clamp_temperature(data: Series, df: DataFrame) -> Series:
    min_temp = -50
    max_temp = 60
    return PandasProcessor(data).clip(min_temp, max_temp).run()


def clamp_wind_speed(data: Series, df: DataFrame) -> Series:
    min_speed = 0
    max_speed = 60  # 60m/s equal to 216 km/h
    return PandasProcessor(data).clip(min_speed, max_speed).run()


DEFAULT_DWD_WEATHER_PROCESSOR_OPTIONS: dict[WeatherDataColumns, ColumnProcessorOptions] = {
    WeatherDataColumns.GH_W_PER_M2: ColumnProcessorOptions(post_processor=solar_irradiance_postprocessor),
    WeatherDataColumns.DH_W_PER_M2: ColumnProcessorOptions(post_processor=diffuse_solar_irradiance_postprocessor),
    WeatherDataColumns.WIND_V_M_PER_S: ColumnProcessorOptions(post_processor=clamp_wind_speed),
    WeatherDataColumns.WIND_DIR_DEGREE: ColumnProcessorOptions(post_processor=wind_dir_postprocessing),
    WeatherDataColumns.T_AIR_DEGREE_CELSIUS: ColumnProcessorOptions(post_processor=clamp_temperature),
}


class DWDWeatherPostProcessor:
    def __init__(
        self,
        processor_options: dict[WeatherDataColumns, ColumnProcessorOptions] = DEFAULT_DWD_WEATHER_PROCESSOR_OPTIONS,
    ) -> None:
        self.processor_options = processor_options

    def __calculate_extra_mappings(self, df: DataFrame):
        for column, processor_options in self.processor_options.items():
            if column in df.columns:
                for extraMapping in processor_options.extra_mappings:
                    print(
                        f"--> calculated data: tar_mean_col: {extraMapping.targetColumn}, src_col: {column.sourceColumn}, fac: {column.unitFactor}"
                    )
                    self._assign_column_data(
                        extraMapping.targetColumn,
                        extraMapping.calculate(df[column.sourceColumn].values, column.unitFactor),
                    )

    def __post_processing(self, df: DataFrame):
        for column, options in self.processor_options.items():
            if column in df.columns:
                if options.post_processor is not None:
                    print(f"Applying post processor on {column}")
                    df[column] = options.post_processor(df[column], df)

    def apply(self, df: DataFrame):
        df_copy = df.copy()
        self.__post_processing(df_copy)
        self.__calculate_extra_mappings(df_copy)
        return df_copy

    def __call__(self, df: DataFrame) -> any:
        return self.apply(df)
