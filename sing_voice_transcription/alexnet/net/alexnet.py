import torch.nn as nn
import torch
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, output_size=52):
        super(AlexNet, self).__init__()
        self.model_name = 'alex'

        # Create model
        self.alexnet = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=False, verbose=False)

        # Modify last linear layer
        num_ftrs = self.alexnet.classifier[-1].in_features
        self.alexnet.classifier[-1] = nn.Linear(num_ftrs, output_size)

        # Modify first conv layer
        num_fout = self.alexnet.features[0].out_channels
        self.alexnet.features[0] = nn.Conv2d(1, num_fout, kernel_size=11, stride=4, padding=2)


    def forward(self, x):
        #print(x.shape)
        #x [50, 1, 1795, 7]
        #x = torch.expand(x, (-1, 1, 359, 35))

        #repeat
        x = x.repeat(1, 1, 1, 10)
        #interpolate nearest
        #x = F.interpolate(x, size = [1975,70], mode='nearest')
        #interpolate bilinear
        #x = F.interpolate(x, size = [1975,70], mode='bilinear')
        out = self.alexnet(x)
        # print(out.shape)
        # [batch, output_size]

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]
        pitch_logits = out[:, 2:]

        return onset_logits, offset_logits, pitch_logits


if __name__ == '__main__':
    from torchsummary import summary
    model = AlexNet().cuda()
    summary(model, input_size=(1, 1795 , 70))
    #summary(model, input_size=(1, 224, 224))
