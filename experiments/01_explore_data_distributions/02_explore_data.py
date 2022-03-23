import numpy as np
from src.data.data_holder import DataHolder
from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter
from src.utils.datetime_utils import dates_to_conditional_vectors


if __name__ == "__main__":
    start_date = DEFAULT_DATA_START_DATE
    end_date = "2009-12-31 23:00:00"
    data_importer = DWDWeatherDataImporter(start_date=start_date, end_date=end_date, auto_preprocess=False)
    data_importer.initialize()
    normalizer = MinMaxNumpyNormalizer()
    normalizer.fit(data_importer.data)

    # data_holder = DataHolder(
    #     data=data_importer.data.values.astype(np.float32),
    #     data_labels=data_importer.get_feature_labels(),
    #     dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
    #     # normalizer_constructor=MinMaxNumpyNormalizer,
    # )

    data_importer_preprocessed = DWDWeatherDataImporter(start_date=start_date, end_date=end_date, auto_preprocess=True)
    data_importer_preprocessed.initialize()
    normalizer_preprocessed = MinMaxNumpyNormalizer()
    normalizer_preprocessed.fit(data_importer_preprocessed.data)

    print("###############################################")
    print("### Raw")
    for column, min, max in zip(data_importer.data.columns, normalizer._data_min, normalizer._data_max):
        print(f"{column}: {min=}, {max=}")

    print("### Preprocessed")
    for column, min, max in zip(
        data_importer_preprocessed.data.columns, normalizer_preprocessed._data_min, normalizer_preprocessed._data_max
    ):
        print(f"{column}: {min=}, {max=}")
