import torch
import torch.nn as nn


class RNNHistogramGenerator(nn.Module):
    def __init__(
            self,
            noise_vector_size: int,
            rnn_layers: int,
            rnn_hidden_size: int,
            gen_features_out: int,
    ):
        super(RNNHistogramGenerator, self).__init__()
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.GRU(noise_vector_size, rnn_hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, gen_features_out)

        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'Generator Parameter Count: {pytorch_total_params}')

    def forward(self, x):
        h0 = torch.zeros(self.rnn_layers, x.size(0), self.rnn_hidden_size).requires_grad_()
        x, hn = self.rnn(x, h0)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    # torch.manual_seed(1)

    BATCH_SIZE = 10
    NOISE_VECTOR_SIZE = 64
    hidden_size = 200
    G_out = 144

    netG = RNNHistogramGenerator(NOISE_VECTOR_SIZE, 1, hidden_size, G_out)
    latent_vector_batch = torch.randn(BATCH_SIZE, 1, NOISE_VECTOR_SIZE)
    print(f'latent_vector_batch: {latent_vector_batch.shape}')
    generated_data = netG(latent_vector_batch)
    print(f'Generated: {generated_data}, \n{generated_data.shape}')
