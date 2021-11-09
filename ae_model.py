import torch
import torch.nn as nn
from torchsummary import summary


class AeNet(nn.Module):
    def __init__(self, size: int, net_list: list = None):
        super().__init__()
        self.__net = []
        self.__keynet = -1

        if len(net_list) % 2 == 0:
            raise UserWarning("net_list参数量必须为奇数")

        __pre = __min = size
        net_list.append(size)
        for i, net_size in enumerate(net_list):
            if i == len(net_list) - 1:
                self.__net.append(
                    nn.Linear(__pre, net_size)
                )
                break
            if i == len(net_list)//2 - 1:
                self.__net.append(
                    nn.Sequential(
                        nn.Linear(__pre, net_size),
                        nn.Sigmoid()
                    )
                )
                self.__keynet=i
            else:
                self.__net.append(
                    torch.nn.Sequential(
                        nn.Linear(__pre, net_size),
                        nn.Tanh()
                    )
                )
            __pre = net_size

        for i in range(0, len(self.__net)):
            setattr(self, 'layer' + str(i), self.__net[i])

    def forward(self, x):
        for layer in self.__net:
            x = layer(x)
        return x

    def __str__(self):
        print(len(self.__net))
        for layer in self.__net:
            print(layer)
        return ''

    def keylayer(self):
        return self.__net[self.__keynet]


if __name__ == '__main__':
    net = AeNet(6, [2, 1, 2])
    print(net, net.keylayer())
    summary(net.cuda(), (6,))
