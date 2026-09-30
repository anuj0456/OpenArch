import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDelayBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same", padding_mode="reflect")
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        return self.activation(self.conv(hidden_states))


class ResNet2Block(nn.Module):
    def __init__(self, in_channels, out_channels, scale, kernel_size, dilation):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channels = out_channels // scale

        self.block = nn.ModuleList([TimeDelayBlock(in_channel, hidden_channels, kernel_size, dilation) for _ in range(scale - 1)])
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_layer in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_layer
            elif i == 1:
                output_part = self.block[i - 1](hidden_layer)
            else:
                output_part = self.block[i - 1](hidden_layer + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1, padding="same", padding_mode="reflect")

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):

        if max_len is None:
            max_len = length.max().long().item()

        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim=dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim=dim))).pow(2).sum(dim=dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        mask = self._length_to_mask(seq_len * lengths, max_len=seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        mask = mask.unsqueeze(1)

        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask/total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_len)
        std = std.unsqueeze(2).repeat(1, 1, seq_len)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=2)

        mean, std = self._compute_statistics(hidden_states, attention)

        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale, se_channel, kernel_size, dilation):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayBlock(in_channels, out_channels, kernel_size, dilation)
        self.res2net_block = ResNet2Block(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayBlock(out_channels, out_channels, kernel_size, dilation)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channel, out_channels)

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.tdnn1(hidden_states)
        hidden_states = self.res2net_block(hidden_states)
        hidden_states = self.tdnn2(hidden_states)
        hidden_states = self.se_block(hidden_states)

        return hidden_states + residual


class Qwen3SpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        self.blocks.append(
            TimeDelayBlock(config.mel_dim, config.enc_channels[0], config.enc_kernel_sizes[0],config.enc_dilations[0])
        )

        for i in range(1, len(config.enc_channels)):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    config.enc_res2net_scale,
                    config.enc_se_channels,
                    config.enc_kernel_sizes[i],
                    config.enc_dilations[i],
                )
            )

        self.mfa = TimeDelayBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        self.asp = AttentiveStatisticsPooling(config.enc_channels[-1], config.enc_attention_channels)

        self.fc = nn.Conv1d(
            config.enc_channels[-1] * 2,
            config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect"
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        for layers in self.blocks:
            hidden_states = layers(hidden_states)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)

        hidden_states = hidden_states.squeeze(-1)
        return hidden_states