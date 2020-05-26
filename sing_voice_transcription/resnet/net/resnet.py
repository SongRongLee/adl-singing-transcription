import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        self.model_name = 'resnet'
        self.resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False, verbose=False, num_classes=num_classes)

    def forward(self, x):
        out = self.resnet(x)
        # print(out.shape)
        # [batch, num_classes]
        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = ResNet().cuda()
    summary(model, input_size=(3, 224, 224))
