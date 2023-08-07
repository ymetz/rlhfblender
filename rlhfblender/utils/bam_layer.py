
import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=4, num_layers=1):
        super(ChannelGate, self).__init__()
        # TODO: figure out it
        # self.gate_activation = gate_activation

        # self.gate_activation = F.relu

        self.gate_c = nn.Sequential()
        self.gate_c.add_module("global_average_pool", nn.AdaptiveAvgPool2d(1))
        self.gate_c.add_module("flatten", nn.Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                "gate_c_fc_%d" % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            # self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module("gate_c_relu_%d" % (i + 1), nn.ReLU())
        self.gate_c.add_module(
            "gate_c_fc_final", nn.Linear(gate_channels[-2], gate_channels[-1])
        )

    def forward(self, in_tensor):
        return self.gate_c(in_tensor).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    # def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
    def __init__(
        self, gate_channel, reduction_ratio=4, dilation_conv_num=2, dilation_val=1
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        # self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            # self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.ReLU())
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
        self.att = None

    def forward(self, in_tensor):
        # in paper authors have a different formula
        # self.att = torch.sigmoid( self.channel_att(in_tensor) + self.spatial_att(in_tensor) )
        self.att = 1 + torch.sigmoid(
            self.channel_att(in_tensor) + self.spatial_att(in_tensor)
        )
        # self.att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )

        return self.att * in_tensor

    def get_attention_map(self):
        return self.att
