from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from statsmodels.distributions import ECDF

from torch.optim.lr_scheduler import StepLR

from plots.ecdf_plot import draw_ecdf_plot, ECDFPlotData
from plots.histogram_plot import HistPlotData, draw_hist_plot
from plots.violin_plot import draw_violin_plot
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
from src.data.synthetical.sinusoidal import generate_sinusoidal_time_series
from src.evaluator.evaluator import Evaluator
from src.gan.discriminator.basic_discriminator import BasicDiscriminator
from src.gan.generator.basic_generator import BasicGenerator
from src.gan.trainer.typing import TrainModel
from src.gan.trainer.vanilla_gan_trainer import VanillaGANTrainer
from src.utils.datetime_utils import convert_input_str_to_date
from src.utils.plot_utils import plot_dfs

if __name__ == '__main__':
    SEED = 1337
    epochs = 100
    noise_vector_size = 50
    sequence_length = 24
    batch_size = 10  # 24
    features = 1
    start_date: str = '2009.01.01'
    end_date: str = '2009.01.31'

    seq_len = 100 * 24
    samples = generate_sinusoidal_time_series(
        sample_count=1,
        series_length=seq_len,
        dimensions=features,
        seed=SEED,
        func=np.sin,
        normalize=False
    )
    input_data = samples[0].astype(np.float32)
    feature_labels = [Feature("Sin value") for i in range(features)]
    data_holder = DataHolder(
        data=input_data,
        data_labels=feature_labels
    )

    G_net = BasicGenerator(input_size=noise_vector_size, out_size=sequence_length * features, hidden_layers=[200, 300, 150])
    G_optim = torch.optim.Adam(G_net.parameters())
    G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    D_net = BasicDiscriminator(input_size=sequence_length * features, out_size=1, hidden_layers=[100, 50, 20])
    D_optim = torch.optim.Adam(D_net.parameters())
    D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    gan_trainer = VanillaGANTrainer(G, D, data_holder, noise_vector_size, sequence_length, batch_size, 'cpu')
    gan_trainer.train(epochs)

    gan_evaluator = Evaluator(gan_trainer.generator.model, feature_labels)

    start = convert_input_str_to_date(start_date)
    end = convert_input_str_to_date(end_date)
    generated_data = gan_evaluator.generate(start, end).numpy().transpose()

    print(f"{generated_data.shape=}")
    data = pd.DataFrame(
        index=pd.date_range(
            start=datetime(start.year, start.month, start.day),
            end=datetime(end.year, end.month, end.day, 23, 59, 59),
            tz="Europe/Berlin",
            freq="H",
        ),
        columns=[f.label for f in feature_labels]
    )
    input_data = input_data.flatten()  # TODO REMOVE
    for i in range(len(feature_labels)):
        gen_data = generated_data[i]
        data[gan_evaluator.feature_labels[i].label] = gen_data

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
            res_pca = visualization(input_data, gen_data, 'pca')
            res_pca.show()
        except Exception as e:
            print(f"Error PCA: {e}")

        try:
            res_tsne = visualization(input_data, gen_data, 'tsne')
            res_tsne.show()
        except Exception as e:
            print(f"Error TSNE: {e}")

        # Draw Box Plots
        box_plot_inputs = [PlotData(input_data, "Original Data"), PlotData(gen_data, "Generated Data")]
        try:
            box_plot_res = draw_box_plot(box_plot_inputs)
            box_plot_res.show()
        except Exception as e:
            print(f"Error BoxPlot: {e}")

        try:
            violin_plot_res = draw_violin_plot(box_plot_inputs)
            violin_plot_res.show()
        except Exception as e:
            print(f"Error Violin Plot: {e}")

        # Draw Histogram
        hist_plot_res = draw_hist_plot([
            HistPlotData(data=input_data, label='Orig'),
            HistPlotData(data=gen_data, label='Generated'),
        ],
            bin_width=0.05
        )
        hist_plot_res.show()

        # Draw ECDF Plot
        draw_ecdf_plot([
            ECDFPlotData(
                data=ECDF(input_data),
                label="Orig data",
                confidence_band_alpha=0.05,
                confidence_band_fill_alpha=0.3,
                confidence_band_label_supplier=lambda alpha: f"{alpha}% confidence band"
            ),
            ECDFPlotData(data=ECDF(gen_data), label="Generated data", confidence_band_alpha=0.00, confidence_band_fill_alpha=0.3),
        ]).show()

        # Draw QQ Plot
        res = draw_qq_plot(
            PlotData(data=input_data, label='Real Values'),
            PlotData(data=gen_data, label='Theoretical quantiles'),
            50,
            {
                # QQReferenceLine.THEORETICAL_LINE,
                QQReferenceLine.FIRST_THIRD_QUARTIL,
                QQReferenceLine.LEAST_SQUARES_REGRESSION
            },
            [0.25, 0.5, 0.75]
        )
        res.fig.show()

    plot_dfs([data])

    for sample_idx in range(len(samples)):
        feature_series = np.transpose(samples[sample_idx])
        for feature_idx in range(len(feature_series)):
            plt.plot(feature_series[feature_idx])
        plt.show()
    # print(samples)


