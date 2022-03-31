from dataclasses import dataclass, field
from typing import Callable, Optional

from pandas import DataFrame
from src.data.weather.weather_dwd_importer import ExtraCalculatedMapping, WeatherDataColumns
from src.data.weather.weather_filters import replace_night_time_values_with_zero, wind_dir_cleansing


@dataclass
class ColumnProcessorOptions:
    extra_mappings: list[ExtraCalculatedMapping] = field(default_factory=list)
    post_processor: Optional[Callable[[DataFrame], DataFrame]] = None


DEFAULT_DWD_WEATHER_PROCESSOR_OPTIONS: dict[WeatherDataColumns, ColumnProcessorOptions] = {
    WeatherDataColumns.GH_W_PER_M2: ColumnProcessorOptions(post_processor=replace_night_time_values_with_zero),
    WeatherDataColumns.DH_W_PER_M2: ColumnProcessorOptions(post_processor=replace_night_time_values_with_zero),
    WeatherDataColumns.WIND_DIR_DEGREE: ColumnProcessorOptions(post_processor=wind_dir_cleansing),
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
                    df[column] = options.post_processor(df[column])

    def apply(self, df: DataFrame):
        df_copy = df.copy()
        self.__post_processing(df_copy)
        self.__calculate_extra_mappings(df_copy)
        return df_copy

    def __call__(self, df: DataFrame) -> any:
        return self.apply(df)
