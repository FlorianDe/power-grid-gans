import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer

"""
The PCA/T-SNE visualization implementation was derived from the following source:
https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/visualization_metrics.py
"""


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    prep_data = ori_data
    prep_data_hat = generated_data
    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        # prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
        tsne_results = tsne.fit_transform(prep_data)
        tsne_results_hat = tsne.fit_transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results_hat[:anal_sample_no, 0], tsne_results_hat[:anal_sample_no, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()


if __name__ == '__main__':
    ori = np.random.rand(20, 6) * 100
    gen = ori + np.random.rand(20, 6) * 5
    snn = StandardNumpyNormalizer()

    snn.fit(ori)
    ori = snn.normalize(ori)

    snn.fit(gen)
    gen = snn.normalize(gen)
    # visualization(ori, gen, 'pca')
    visualization(ori, gen, 'tsne')
