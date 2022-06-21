import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution Module
class Conv_block(nn.Module):
      def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.Lrelu2 = nn.LeakyReLU(True)
        
        # When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the input. Sliding windows that would start in the right padded region are ignored.
        self.mp = nn.MaxPool2d(kernel_size=pooling, stride=pooling, ceil_mode=True) 
        #self.bn = nn.BatchNorm2d(output_channels)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
         
      def forward(self, x):
        
        x = self.Lrelu1(self.bn1(self.conv1(x)))
        out = self.mp(self.Lrelu2(self.bn2(self.conv2(x))))
        
        return out
    
class Encoder_module(nn.Module):
        def __init__(self, IsVAE=False):
            super(Encoder_module, self).__init__()
            #  input sample of size  69 × 240 (x 1) - BCHW B x 1 x 69 × 240 
            #  resized by pooling, not conv
            self.Conv_block1 = Conv_block(input_channels = 1, output_channels = 32, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block2 = Conv_block(input_channels = 32, output_channels = 64, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block3 = Conv_block(input_channels = 64, output_channels = 128, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block4 = Conv_block(input_channels = 128, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            self.Conv_block5 = Conv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
            # output latent size 3 × 8 × 256  - HWC B x 256 x 3 × 8 
            self.IsVAE = IsVAE
            if self.IsVAE ==True:
                self.Conv_block_std = Conv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=1, padding=1, pooling=2)
        def forward(self, x):
            x = self.Conv_block1(x)
            x = self.Conv_block2(x)
            x = self.Conv_block3(x)
            x = self.Conv_block4(x)
            out = self.Conv_block5(x)
            if self.IsVAE == True:
                logvar = self.Conv_block_std(x)
                return out , logvar
            return out

class DeConv_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding = 0 ):
        super(DeConv_block, self).__init__()
        self.ConvTrans1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding) # upsample
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.Lrelu1 = nn.LeakyReLU(True)
        
        self.ConvTrans2 = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding) # no sizing
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.Lrelu2 = nn.LeakyReLU(True)
        
        
        nn.init.xavier_normal_(self.ConvTrans1.weight)
        nn.init.xavier_normal_(self.ConvTrans2.weight)
        #self.BN = nn.BatchNorm2d(OutChannel)
    def forward(self, x):
        x = self.Lrelu1(self.bn1(self.ConvTrans1(x)))
        #x = self.Lrelu1(self.ConvTrans1(x))
        out = self.Lrelu2( self.bn2(self.ConvTrans2(x)))
        #out = self.Lrelu2(self.ConvTrans2(x))
        return out
    
    
class Decoder_module(nn.Module):
        def __init__(self):
            super(Decoder_module, self).__init__()
            # input latent size 3 × 8 × 256  - HWC
            self.DeConv_block1 = DeConv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=2, padding=1, output_padding=0)
            self.DeConv_block2 = DeConv_block(input_channels = 256, output_channels = 256, kernel_size=3, stride=2, padding=1, output_padding=(0,1))
            self.DeConv_block3 = DeConv_block(input_channels = 256, output_channels = 128, kernel_size=3, stride=2, padding=1, output_padding=(1,1))
            self.DeConv_block4 = DeConv_block(input_channels = 128, output_channels = 64, kernel_size=3, stride=2, padding=1, output_padding=(0,1))
            #self.DeConv_block5 = DeConv_block(input_channels = output_channels, output_channels = output_channels, kernel_size=3, stride=1, padding=1, pooling=1)
            self.ConvTrans_last2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False, output_padding=(0,1))
            self.Lrelu = nn.LeakyReLU(True)
            self.ConvTrans_last = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
            #  output of size  69 × 240 (x 1) - HWC
        def forward(self, x):
            x = self.DeConv_block1(x)
            x = self.DeConv_block2(x)
            x = self.DeConv_block3(x)
            x = self.DeConv_block4(x)
            x = self.Lrelu(self.ConvTrans_last2(x))
            out = self.ConvTrans_last(x) # no acivation at last
            return out
        
        
class Convolutional_AE(nn.Module):
    def __init__(self):
        super(Convolutional_AE, self).__init__()
        # input sample of size 69 × 240
        self.Incoder_module = Encoder_module()
        self.Decoder_module = Decoder_module()

    def forward(self, x):
        latent = self.Incoder_module(x)
        out = self.Decoder_module(latent)
        return out
    
class Convolutional_VAE(nn.Module):
    def __init__(self):
        super(Convolutional_VAE, self).__init__()
        # input sample of size 69 × 240
        self.Incoder_module = Encoder_module(IsVAE=True)
        self.Decoder_module = Decoder_module()
        
    def sampling(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean
    
    def forward(self, x):
        mean, logvar = self.Incoder_module(x)
        latent = self.sampling(mean, logvar)
        out = self.Decoder_module(latent)
        return out
 
    
if __name__ == '__main__':
        print("##Size Check")
        
        '''print("##Encoding##")
        input = torch.randn(32, 1, 69, 240)
        print("input: ", input.shape)
        p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        #p2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)
        output = p(input)
        print("output1: ", output.shape)
        output = p(output)
        print("output2: ", output.shape)
        output = p(output)
        print("output3: ", output.shape)
        output = p(output)
        print("output4: ", output.shape)
        output = p(output)
        print("output5: ", output.shape)'''
        
        
        print("##Decoding##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding
        m = nn.ConvTranspose2d(32, 32, 3, 2, 1)
        #m2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        # 3, 8  / 2
        m3 = nn.ConvTranspose2d(32, 32, 3, 2, 1, (1,1))
        m4 = nn.ConvTranspose2d(32, 32, 3, 2, 1, (0,1))
        output = m(input)
        print("output1: ", output.shape)
        output = m4(output)
        print("output2: ", output.shape)
        output = m3(output)
        print("output3: ", output.shape)
        output = m4(output)
        print("output4: ", output.shape)
        output = m4(output)
        print("output5: ", output.shape)
        
        '''print("##Decoding 2 same with convtrans##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        m = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        output = m(input)
        print("output: ", output.shape)'''
        
        
        
        print("##Decodin up sampleing##")
        input = torch.randn(32, 32, 3, 8)
        print("input: ", input.shape)
        #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding
        up = nn.Upsample(size =(5,15), mode='nearest')
        up2 = nn.Upsample(size=(9,30), mode='nearest')
        up3 = nn.Upsample(size=(18, 60), mode='nearest')
        up4 = nn.Upsample(size=(35, 120), mode='nearest')
        up5 = nn.Upsample(size=(69, 240), mode='nearest')
        #m2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        # 3, 8  / 2

        output = up(input)
        print("output1: ", output.shape)
        output = up2(output)
        print("output2: ", output.shape)
        output = up3(output)
        print("output3: ", output.shape)
        output = up4(output)
        print("output4: ", output.shape)
        output = up5(output)
        print("output5: ", output.shape)
