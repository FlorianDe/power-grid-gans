import math
import warnings

import seaborn as sns

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from statsmodels.distributions import ECDF
from datetime import datetime

from torch.optim.lr_scheduler import StepLR

from experiments.utils import get_experiments_folder
from src.gan.discriminator.basic_discriminator import BasicDiscriminator
from src.gan.generator.basic_generator import BasicGenerator
from src.gan.trainer.vanilla_gan_trainer import VanillaGANTrainer
from src.gan.discriminator.cnn_discriminator import CNNDiscriminator
from src.gan.generator.cnn_generator import CNNGenerator
from src.gan.trainer.cgan_trainer import CGANTrainer
from src.plots.ecdf_plot import draw_ecdf_plot, ECDFPlotData
from src.plots.histogram_plot import HistPlotData, draw_hist_plot
from src.plots.violin_plot import draw_violin_plot
from src.plots.box_plot import draw_box_plot
from src.plots.qq_plot import draw_qq_plot, QQReferenceLine
from src.plots.typing import PlotData
from src.metrics.jensen_shannon import js_divergence, js_distance
from src.metrics.kullback_leibler import kl_divergence
from src.metrics.pca import visualization
from src.metrics.r_squared import r_squared
from src.metrics.kolmogorov_smirnov import ks2_test, ks2_critical_value
from src.data.typing import Feature
from src.data.data_holder import DataHolder
from src.data.synthetical.sinusoidal import generate_sinusoidal_time_series, TrigFuncParameters
from src.evaluator.evaluator import Evaluator
from src.gan.trainer.typing import TrainModel
from src.utils.datetime_utils import convert_input_str_to_date, dates_to_conditional_vectors
from src.utils.pandas_utils import get_datetime_values


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    warnings.warn("Have been overhauled by 02_01_vanila_gan_sines.py", DeprecationWarning, stacklevel=2)
    sns.set_theme()
    sns.set_context("paper")
    sinusoidal_dists_root_folder = get_experiments_folder().joinpath("02_sinusoidal_data_gans")
    experiment_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_folder = sinusoidal_dists_root_folder / experiment_folder_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    plot_extension = ".pdf"

    SEED = 1337
    epochs = 1000
    noise_vector_size = 1
    sequence_length = 24
    samples_len = sequence_length * 7
    batch_size = 5
    features = 1
    start_date: str = "2020.01.01"
    end_date: str = "2020.01.01"

    samples = generate_sinusoidal_time_series(
        sample_count=1,
        series_length=samples_len,
        dimensions=features,
        seed=SEED,
        func=np.sin,
        normalize=False,
        trig_parameters=TrigFuncParameters(2 * math.pi / 24, 0, 1),
    )
    input_data = samples[0].astype(np.float32)
    feature_labels = [Feature("Sin value") for i in range(features)]
    dates = pd.date_range(start="01/01/2021", freq="h", periods=samples_len)
    data_holder = DataHolder(
        data=np.array(list(chunks(input_data, sequence_length))),
        data_labels=feature_labels,
        dates=np.array(
            list(chunks(np.array(dates_to_conditional_vectors(*get_datetime_values(dates))), sequence_length))
        ),
    )

    # G_net = CNNGenerator(input_size=noise_vector_size, out_size=features)
    G_net = BasicGenerator(
        input_size=noise_vector_size, out_size=sequence_length * features, hidden_layers=[256, 512, 1024, 512]
    )
    G_optim = torch.optim.Adam(G_net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    G_sched = None  # StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    # D_net = CNNDiscriminator(input_size=features, out_size=1)
    D_net = BasicDiscriminator(input_size=sequence_length * features, out_size=1, hidden_layers=[1024, 512, 256])
    D_optim = torch.optim.Adam(D_net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    D_sched = None  # StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    # summary(G_net, (features, noise_vector_size))
    # summary(D_net, (features, sequence_length))

    gan_trainer = VanillaGANTrainer(
        generator=G,
        discriminator=D,
        data_holder=data_holder,
        noise_vector_size=noise_vector_size,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device="cpu",
    )
    gan_trainer.train(epochs)

    # G_net = BasicGenerator(input_size=noise_vector_size + 14, out_size=features, hidden_layers=[256, 512, 1024, 512])
    # G_optim = torch.optim.Adam(G_net.parameters(), lr=0.003, betas=(0.9, 0.999))
    # G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    # G = TrainModel(G_net, G_optim, G_sched)
    #
    # D_net = BasicDiscriminator(input_size=features + 14, out_size=1, hidden_layers=[1024, 512, 256])
    # D_optim = torch.optim.Adam(D_net.parameters(), lr=0.003, betas=(0.9, 0.999))
    # D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    # D = TrainModel(D_net, D_optim, D_sched)
    #
    # gan_trainer = CGANTrainer(G, D, data_holder, noise_vector_size, batch_size, 'cpu')
    # gan_trainer.train(epochs)

    gan_evaluator = Evaluator(gan_trainer.generator.model, noise_vector_size, feature_labels)

    start = convert_input_str_to_date(start_date)
    end = convert_input_str_to_date(end_date)
    generated_data = gan_evaluator.generate(start, end, with_conditions=False)

    generated_data_first_batch = generated_data[0].detach().numpy()
    print(f"{generated_data.shape=}")
    print(f"{generated_data_first_batch.shape=}")
    # data = pd.DataFrame(
    #     index=pd.date_range(
    #         start=datetime(start.year, start.month, start.day),
    #         end=datetime(end.year, end.month, end.day, 23, 59, 59),
    #         tz="Europe/Berlin",
    #         freq="H",
    #     ),
    #     columns=[f.label for f in feature_labels]
    # )
    input_data = input_data.flatten()  # TODO REMOVE
    for i in range(len(feature_labels)):
        gen_data = generated_data_first_batch[i]
        # data[gan_evaluator.feature_labels[i].label] = gen_data

        ### Metrics on PDFs #TODO SAME BUCKETS SOMEHOW
        # R² Metric
        try:
            r_squared_res = r_squared(input_data, gen_data)
            print(f"R² (R squared): {r_squared_res}")
        except Exception as e:
            print(f"Error R_Squared: {e}")

        # KL-Metric
        try:
            kl_div_res = kl_divergence(input_data, gen_data)
            print(f"KL-Divergence: {kl_div_res}")
        except Exception as e:
            print(f"Error KL-Divergence: {e}")

        # JS-Metrics (Div/Dist)
        try:
            js_div_res = js_divergence(input_data, gen_data)
            print(f"JS-Divergence: {js_div_res}")
            js_dist_res = js_distance(input_data, gen_data)
            print(f"JS-Distance: {js_dist_res}")
        except Exception as e:
            print(f"Error JS-Div/Dist: {e}")

        # KS TEST
        try:
            ks_res, ks_p_value = ks2_test(input_data, gen_data)
            ks_crit = ks2_critical_value(input_data, gen_data, alpha=0.05)
            print(f"KS Test results {ks_res=}, {ks_p_value=}, {ks_crit=}")
            print(f"KS Test {'proved' if ks_crit < ks_res else 'disproved'} the 0-hypothesis.")
        except Exception as e:
            print(f"Error KS-Test: {e}")

        ### Visualizations and plots
        # PCA and T-SNE PLOTS
        try:
            res_pca = visualization(input_data, gen_data, "pca")
            res_pca.show()
        except Exception as e:
            print(f"Error PCA: {e}")

        try:
            res_tsne = visualization(input_data, gen_data, "tsne")
            res_tsne.show()
        except Exception as e:
            print(f"Error TSNE: {e}")

        # Draw Box Plots
        box_plot_inputs = [PlotData(input_data, "Original Data"), PlotData(gen_data, "Generated Data")]
        try:
            box_plot_res = draw_box_plot(box_plot_inputs)
            box_plot_res.fig.savefig(experiment_folder / f"box_plot{plot_extension}")
            box_plot_res.show()
        except Exception as e:
            print(f"Error BoxPlot: {e}")

        try:
            violin_plot_res = draw_violin_plot(box_plot_inputs)
            violin_plot_res.fig.savefig(experiment_folder / f"violin_plot{plot_extension}")
            violin_plot_res.show()
        except Exception as e:
            print(f"Error Violin Plot: {e}")

        # Draw Histogram
        hist_plot_res = draw_hist_plot(
            [
                HistPlotData(data=input_data, label="Orig"),
                HistPlotData(data=gen_data, label="Generated"),
            ],
            bin_width=0.05,
        )
        hist_plot_res.fig.savefig(experiment_folder / f"histogram_plot{plot_extension}")
        hist_plot_res.show()

        # Draw ECDF Plot
        ecdf_plot_res = draw_ecdf_plot(
            [
                ECDFPlotData(
                    data=ECDF(input_data),
                    label="Orig data",
                    confidence_band_alpha=0.05,
                    confidence_band_fill_alpha=0.3,
                    confidence_band_label_supplier=lambda alpha: f"{alpha}% confidence band",
                ),
                ECDFPlotData(
                    data=ECDF(gen_data),
                    label="Generated data",
                    confidence_band_alpha=0.00,
                    confidence_band_fill_alpha=0.3,
                ),
            ]
        )
        ecdf_plot_res.fig.savefig(experiment_folder / f"ecdf_plot{plot_extension}")
        ecdf_plot_res.show()

        # Draw QQ Plot
        qq_plot_res = draw_qq_plot(
            PlotData(data=input_data, label="Real Values"),
            PlotData(data=gen_data, label="Theoretical quantiles"),
            50,
            {
                # QQReferenceLine.THEORETICAL_LINE,
                QQReferenceLine.FIRST_THIRD_QUARTIL,
                QQReferenceLine.LEAST_SQUARES_REGRESSION,
            },
            [0.25, 0.5, 0.75],
        )
        qq_plot_res.fig.savefig(experiment_folder / f"qq_plot{plot_extension}")
        qq_plot_res.fig.show()

    # plot_dfs([data])

    for feature_idx in range(features):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(generated_data_first_batch[feature_idx])
        ax.plot(samples[feature_idx])
        fig.savefig(experiment_folder / f"comparison_{feature_idx}{plot_extension}")
        fig.show()

    # for sample_idx in range(len(samples)):
    #     feature_series = np.transpose(samples[sample_idx])
    #     for feature_idx in range(len(feature_series)):
    #         fig, ax = plt.subplots(nrows=1, ncols=1)
    #         ax.plot(feature_series[feature_idx])
    #         fig.show()
    # print(samples)
