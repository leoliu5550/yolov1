import torch 
import torch.nn as nn

architecture_config = [
    #Conv (kernl_size,out_put,stride,padding)
    (7, 64, 2, 3),
    "M",#MaxPooling (kernl_size =2 ,stride = 2)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #[conv,Conv,repeat_times]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

class Yolov1(nn.Module):
    def __init__(self,in_channels = 3,**kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self,x):
        x = self.darknet(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self,architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size =x[0],
                        stride = x[2],
                        padding = x[3])]
                in_channels = x[1]

            elif type(x) == str:
                layers+=[
                    nn.MaxPool2d(kernel_size=(2,2),stride=2)
                    ]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers +=[
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size = conv1[0],
                            stride=conv1[2],
                            padding = conv1[3]
                        )]
                    layers +=[
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self,split_size,num_boxes,num_classes):
        S = split_size
        B = num_boxes
        C = num_classes
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S,1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1024,S * S *(C+B*5)),
        )
        return model

