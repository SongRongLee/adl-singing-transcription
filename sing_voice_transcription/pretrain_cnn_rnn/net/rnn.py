import torch.nn as nn
import torch
from pathlib import Path

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from func2module import Interpolate, Flatten
from resnet.net.resnet import ResNet
from efficientnet.net.effnet import EffNetb0
from alexnet.net.alexnet import AlexNet


class RNN(nn.Module):
    def __init__(self, pretrain_cnn_path_list, output_size=52, hidden_size=128):
        super(RNN, self).__init__()
        self.model_name = 'rnn'

        assert any(pretrain_cnn_path_list)
        assert all([Path(path).exists() for path in pretrain_cnn_path_list if path])
        
        self.feature_size_table = {'resnet': 2048, 'effent': 1280, 'alexnet': 4096}
        self.embedding_dim = 0

        # Create feature extractor of pretrain models
        self.feature_extractor_list = []
        for idx, path in enumerate(pretrain_cnn_path_list):
            if path == None: continue
            
            # Exclude the last module from original models
            if idx == 0:
                model = ResNet().cuda()
                feature_extractor = torch.nn.Sequential(*list(model.resnet.children())[:-1])
                self.embedding_dim += self.feature_size_table['resnet']
            elif idx == 1:
                model = EffNetb0().cuda()
                feature_extractor = torch.nn.Sequential(*list(model.effnet.children())[:-1])
                self.embedding_dim += self.feature_size_table['effent']
            elif idx == 2:
                model = AlexNet().cuda()
                bilinear_layer = Interpolate(size=[1975, 70], mode='bilinear')
                flatten_layer = Flatten()
                feature_extractor = torch.nn.Sequential(bilinear_layer,
                                                        *list(model.alexnet.children())[:-1],
                                                        flatten_layer,
                                                        torch.nn.Sequential(*list(model.alexnet.classifier.children())[:4]))
                self.embedding_dim += self.feature_size_table['alexnet']
                
            model.load_state_dict(torch.load(path))

            if torch.cuda.device_count() > 1:
                feature_extractor = nn.DataParallel(feature_extractor)

            # Disable gradient
            for param in feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor_list += [feature_extractor]

        # Create model
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,     
            hidden_size=hidden_size,   
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Define last linear layer
        self.out = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        # print(x.shape)
        # [batch_size, chunk_size, channel_num, feature_size, window_size]

        batch_size, chunk_size, _, _, _ = x.shape

        chunk_feature = torch.empty(batch_size, chunk_size, self.embedding_dim)

        for chunk_idx in range(chunk_size):
            frame_feature = [fe(x[:, chunk_idx]).squeeze(-1).squeeze(-1) for fe in self.feature_extractor_list]
            frame_feature = torch.cat(frame_feature, dim=1)
            chunk_feature[:, chunk_idx] = frame_feature
        
        # ResNet [50, 150, 2048, 1, 1]
        # EffNet [50, 150, 1280, 1, 1]
        # EffNet [50, 150, 4096, 1, 1]

        self.rnn.flatten_parameters()
        r_out, (h_n, h_c) = self.rnn(chunk_feature.cuda(), None) 
        # print(r_out.shape)
        # [batch_size, chunk_size, lstm_hidden_size]
        
        out = self.out(r_out)
        
        onset_logits = out[:, :, 0]
        offset_logits = out[:, :, 1]
        pitch_logits = out[:, :, 2:]

        return onset_logits, offset_logits, pitch_logits


if __name__ == '__main__':
    from torchsummary import summary
    batch_size, chunk_size, feature_size, window_size = 5, 150, 1795, 7
    model = RNN(pretrain_cnn_path_list=["../resnet/models/resnet_model",
                                        "../efficientnet/models/effnet_e_8000"]).cuda()
    summary(model, input_data=(chunk_size, 1, feature_size, window_size), device="cuda")
