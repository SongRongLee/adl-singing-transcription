import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, output_size=52):
        super(ResNet, self).__init__()
        self.model_name = 'resnet'

        # Create model
        self.resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=False, verbose=False)

        # Modify last linear layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_size)

        # Modify first conv layer
        num_fout = self.resnet.conv1.out_channels
        self.resnet.conv1 = nn.Conv2d(1, num_fout, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        out = self.resnet(x)
        # print(out.shape)
        # [batch, output_size]

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]
        pitch_logits = out[:, 2:]

        return onset_logits, offset_logits, pitch_logits


if __name__ == '__main__':
    from torchsummary import summary
    model = ResNet().cuda()
    summary(model, input_size=(1, 224, 224))
