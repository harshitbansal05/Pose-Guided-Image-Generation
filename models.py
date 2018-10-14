import torch
import torch.optim as optim
import torch.nn as nn

from torch import cat
import numpy as np

class GeneratorCNN_Pose_UAEAfterResidual_256(nn.Module):
    
    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
        
    def fc(self, ch_in, ch_out):
        return nn.Linear(ch_in, ch_out)
    
    def __init__(self, ch_in, z_num, repeat_num, hidden_num=128):
        super(GeneratorCNN_Pose_UAEAfterResidual_256, self).__init__()
        self.min_fea_map_H = 8
        self.z_num = z_num 
        self.hidden_num = hidden_num 
        self.repeat_num = repeat_num
        
        self.block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)

        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(256, 256, 3, 1)
        self.block3 = self.block(384, 384, 3, 1)
        self.block4 = self.block(512, 512, 3, 1)
        self.block5 = self.block(640, 640, 3, 1)
        self.block6 = self.block(768, 768, 3, 1)
            
        self.block_one1 = self.block_one(128, 256, 3, 2)
        self.block_one2 = self.block_one(256, 384, 3, 2)
        self.block_one3 = self.block_one(384, 512, 3, 2)
        self.block_one4 = self.block_one(512, 640, 3, 2)
        self.block_one5 = self.block_one(640, 768, 3, 2)
        
        self.fc1 = self.fc(self.min_fea_map_H * self.min_fea_map_H * 768, self.z_num)
        self.fc2 = self.fc(self.z_num, self.min_fea_map_H * self.min_fea_map_H * self.hidden_num)
        
        self.block7 = self.block(896, 896, 3, 1)
        self.block8 = self.block(1280, 1280, 3, 1)
        self.block9 = self.block(1024, 1024, 3, 1)
        self.block10 = self.block(768, 768, 3, 1)
        self.block11 = self.block(512, 512, 3, 1)
        self.block12 = self.block(256, 256, 3, 1)
        
        self.block_one6 = self.block_one(896, 640, 1, 1, padding=0)
        self.block_one7 = self.block_one(1280, 512, 1, 1, padding=0)
        self.block_one8 = self.block_one(1024, 384, 1, 1, padding=0)
        self.block_one9 = self.block_one(768, 256, 1, 1, padding=0)
        self.block_one10 = self.block_one(512, 128, 1, 1, padding=0)
        
        self.conv_last = self.conv(256, 3, 3, 1) 
        
        self.upscale = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        encoder_layer_list = []
        
        x = self.block_1(x)
        
        # 1st encoding layer
        res = x
        x = self.block1(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # 2nd encoding layer
        res = x
        x = self.block2(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # 3rd encoding layer
        res = x
        x = self.block3(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # 4th encoding layer
        res = x
        x = self.block4(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one4(x)
        # 5th encoding layer
        res = x
        x = self.block5(x)
        x = x + res
        encoder_layer_list.append(x)
        x = self.block_one5(x)
        # 6th encoding layer
        res = x
        x = self.block6(x)
        x = x + res
        encoder_layer_list.append(x)
            
        x = x.view(-1, self.min_fea_map_H * self.min_fea_map_H * 768)
        x = self.fc1(x)
        z = x
        
        x = self.fc2(z)
        x = x.view(-1, self.hidden_num, self.min_fea_map_H, self.min_fea_map_H)
        
        # 1st decoding layer
        x = torch.cat([x, encoder_layer_list[5]], dim=1)
        res = x
        x = self.block7(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one6(x)
        # 2nd decoding layer
        x = torch.cat([x, encoder_layer_list[4]], dim=1)
        res = x
        x = self.block8(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one7(x)
        # 3rd decoding layer
        x = torch.cat([x, encoder_layer_list[3]], dim=1)
        res = x
        x = self.block9(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one8(x)
        # 4th decoding layer
        x = torch.cat([x, encoder_layer_list[2]], dim=1)
        res = x
        x = self.block10(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one9(x)
        # 5th decoding layer
        x = torch.cat([x, encoder_layer_list[1]], dim=1)
        res = x
        x = self.block11(x)
        x = x + res
        x = self.upscale(x)
        x = self.block_one10(x)
        # 6th decoding layer
        x = torch.cat([x, encoder_layer_list[0]], dim=1)
        res = x
        x = self.block12(x)
        x = x + res
       
        output = self.conv_last(x)
        return output


class UAE_noFC_AfterNoise(nn.Module):
    
    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
        
    def __init__(self, ch_in, repeat_num, hidden_num=128):
        super(UAE_noFC_AfterNoise, self).__init__()
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num
        
        self.block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)
        
        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(128, 256, 3, 1)
        self.block3 = self.block(256, 384, 3, 1)
        self.block4 = self.block(384, 512, 3, 1)
            
        self.block_one1 = self.block_one(128, 128, 3, 2)
        self.block_one2 = self.block_one(256, 256, 3, 2)
        self.block_one3 = self.block_one(384, 384, 3, 2)
        
        self.block5 = self.block(1024, 128, 3, 1)
        self.block6 = self.block(512, 128, 3, 1)
        self.block7 = self.block(384, 128, 3, 1)
        self.block8 = self.block(256, 128, 3, 1)
        
        self.conv_last = self.conv(128, 3, 3, 1)
        
        self.upscale = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        encoder_layer_list = []
        
        x = self.block_1(x)
        
        # 1st encoding layer
        x = self.block1(x)
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # 2nd encoding layer
        x = self.block2(x)
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # 3rd encoding layer
        x = self.block3(x)
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # 4th encoding layer
        x = self.block4(x)
        encoder_layer_list.append(x)
        
        # 1st decoding layer
        x = torch.cat([x, encoder_layer_list[-1]], dim=1)
        x = self.block5(x)
        x = self.upscale(x)
        # 2nd decoding layer
        x = torch.cat([x, encoder_layer_list[-2]], dim=1)
        x = self.block6(x)
        x = self.upscale(x)
        # 3rd decoding layer
        x = torch.cat([x, encoder_layer_list[-3]], dim=1)
        x = self.block7(x)
        x = self.upscale(x)
        # 4th decoding layer
        x = torch.cat([x, encoder_layer_list[-4]], dim=1)
        x = self.block8(x)
        
        output = self.conv_last(x)
        return output


class DCGANDiscriminator_256(nn.Module):

    def uniform(self, stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')
    
    def LeakyReLU(self, x, alpha=0.2):
        return torch.max(alpha*x, x)

    def conv2d(self, x, input_dim, filter_size, output_dim, gain=1, stride=1, padding=2):
        filter_values = self.uniform(
                self._weights_stdev,
                (output_dim, input_dim, filter_size, filter_size)
            )
        filter_values *= gain
        filters = torch.from_numpy(filter_values)
        biases = torch.from_numpy(np.zeros(output_dim, dtype='float32'))
        if self.use_gpu:
            filters = filters.cuda()
            biases = biases.cuda()
        result = nn.functional.conv2d(x, filters, biases, stride, padding)
        return result
        
    def LayerNorm(self, ch):
        return nn.BatchNorm2d(ch)
        
    def __init__(self, bn=True, input_dim=3, dim=64, _weights_stdev=0.02, use_gpu=True):
        super(DCGANDiscriminator_256, self).__init__()
        self.bn = bn
        self.input_dim = input_dim
        self.dim = dim
        self._weights_stdev = _weights_stdev
        self.use_gpu = use_gpu

        self.bn1 = self.LayerNorm(2*self.dim)
        self.bn2 = self.LayerNorm(4*self.dim)
        self.bn3 = self.LayerNorm(8*self.dim)
        self.bn4 = self.LayerNorm(8*self.dim)
        
        self.fc1 = nn.Linear(8*8*8*self.dim, 1)
        
    def forward(self, x):
        output = x
        
        output = self.conv2d(output, self.input_dim, 5, self.dim, stride=2)
        output = self.LeakyReLU(output)
        
        output = self.conv2d(output, self.dim, 5, 2*self.dim, stride=2)
        if self.bn:
            output = self.bn1(output)
        output = self.LeakyReLU(output)
        
        output = self.conv2d(output, 2*self.dim, 5, 4*self.dim, stride=2)
        if self.bn:
            output = self.bn2(output)
        output = self.LeakyReLU(output)
        
        output = self.conv2d(output, 4*self.dim, 5, 8*self.dim, stride=2)
        if self.bn:
            output = self.bn3(output)
        output = self.LeakyReLU(output)
        
        output = self.conv2d(output, 8*self.dim, 5, 8*self.dim, stride=2)
        if self.bn:
            output = self.bn4(output)
        output = self.LeakyReLU(output)
        
        output = output.view(-1, 8*8*8*self.dim)
        output = self.fc1(output)
        return output
