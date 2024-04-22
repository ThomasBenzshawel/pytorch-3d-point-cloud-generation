"""Build Point Cloud Generator Pytorch model"""
import torch
from torch import nn
from torch.nn import functional as F


def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def deconv2d_block_trans(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 3, stride=2,
                           padding=1, output_padding=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

def pixel_bias(outViewN, outW, outH, renderDepth):
    X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
    X, Y = X.float(), Y.float() # [H,W]
    initTile = torch.cat([
        X.repeat([outViewN, 1, 1]), # [V,H,W]
        Y.repeat([outViewN, 1, 1]), # [V,H,W]
        torch.ones([outViewN, outH, outW]).float() * renderDepth, 
        torch.zeros([outViewN, outH, outW]).float(),
    ], dim=0) # [4V,H,W]

    return initTile.unsqueeze_(dim=0) # [1,4V,H,W]


class U_Net(nn.Module):
    """Encoder of Structure Generator"""
    # def __init__(self):
    #     super(Encoder, self).__init__()
    #     self.conv1 = conv2d_block(3, 96)
    #     self.conv2 = conv2d_block(96, 128)
    #     self.conv3 = conv2d_block(128, 192)
    #     self.conv4 = conv2d_block(192, 256)
    #     self.fc1 = linear_block(4096, 2048) # After flatten
    #     self.fc2 = linear_block(2048, 1024)
    #     self.fc3 = nn.Linear(1024, 512)


    ## Experimenting with larger encoder
    def __init__(self, outViewN, outW, outH, renderDepth):
        super(U_Net, self).__init__()
        self.outViewN = outViewN
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #input is 64x64x3
        self.conv11e = conv2d_block(3, 64) # size 64x64x64
        self.conv12e = conv2d_block(64, 64)# size 64x64x64
        # pool size 32x32x64

        self.conv21e = conv2d_block(64, 128) # size 
        self.conv22e = conv2d_block(128, 128) # size 
        # pool size 16x16x128

        self.conv31e = conv2d_block(128, 256)  
        self.conv32e = conv2d_block(256, 256) 
        # pool size 8x8x256

        self.conv41e = conv2d_block(256, 512)
        self.conv42e = conv2d_block(512, 512)
        # pool size 4x4x512

        #Latent space
        self.fc1e = linear_block(16384, 8192) # After flatten 8192
        self.fc2e = linear_block(8192, 4096) # After flatten 8192
        self.fc3e = linear_block(4096, 2048)

        self.fc4e = nn.Linear(2048, 1024)

        self.relu = nn.ReLU()

        self.fc1d = linear_block(1024, 2048)
        self.fc2d = linear_block(2048, 4096)

        #Decoder

        self.deconv1 = deconv2d_block_trans(256, 256)
        self.convd11 = deconv2d_block(256, 256)

        self.deconv2 = deconv2d_block_trans(256, 128)
        self.convd21 = deconv2d_block(128, 128)

        self.deconv3 = deconv2d_block_trans(128, 64)
        self.convd31 = deconv2d_block(64, 64)

        self.deconv5 = deconv2d_block_trans(64, 48)
        self.convd51 = deconv2d_block(48, 48)

        self.pixel_conv = nn.Conv2d(48, outViewN*4, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        x11e = self.conv11e(x)
        # print(x11e.shape, "x11e")
        x12e = self.conv12e(x11e)
        # print(x12e.shape, "x12e")
        x1p = self.pool(x12e)
        # print(x1p.shape, "x1p")

        x21e = self.conv21e(x1p)
        # print(x21e.shape, "x21e")
        x22e = self.conv22e(x21e)
        # print(x22e.shape, "x22e")
        x2p = self.pool(x22e)
        # print(x2p.shape, "x2p")

        x31e = self.conv31e(x2p)
        # print(x31e.shape, "x31e")
        x32e = self.conv32e(x31e)
        # print(x32e.shape, "x32e")
        x3p = self.pool(x32e)
        # print(x3p.shape, "x3p")

        x4e = self.fc1e(x3p.contiguous().view(-1, 16384))
        # print(x4e.shape, "x4e")
        x5e = self.fc2e(x4e)
        # print(x5e.shape, "x5e")
        x6e = self.fc3e(x5e)
        # print(x6e.shape, "x6e")
        x7e = self.fc4e(x6e)

        x6e = self.relu(x7e)

        x1d = self.fc1d(x6e)
        # print(x1d.shape, "x1d")
        x2d = self.fc2d(x1d)
        # print(x2d.shape, "x2d")

        x4d = self.deconv1(x2d.view(-1, 256, 4, 4))
        # print(x4d.shape, "x4d")
        x5d = self.convd11(x4d + x3p)
        # print(x5d.shape, "x5d")
        x6d = self.deconv2(x5d)
        # print(x6d.shape, "x6d")
        
        x7d = self.convd21(x6d + x2p)
        # print(x7d.shape, "x7d")
        x8d = self.deconv3(x7d)
        # print(x8d.shape, "x8d")
        x9d = self.convd31(x8d + x1p)


        x10d = self.deconv5(x9d)
        x11d = self.convd51(F.interpolate(x10d, scale_factor=2))

        x12d = self.pixel_conv(x11d) 
        x13d = self.pixel_bias.to(x12d.device)

        x14d = x12d + x13d
        
        XYZ, maskLogit = torch.split(
            x14d, [self.outViewN * 3, self.outViewN], dim=1)

        return XYZ, maskLogit




class Structure_Generator_Unet(nn.Module):
    """Structure generator components in PCG"""

    def __init__(self, encoder=None, decoder=None,
                 outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(Structure_Generator_Unet, self).__init__()

        self.U_Net = U_Net(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        return self.U_Net(x)


# # TESTING
# if __name__ == '__main__':
#     import options
#     cfg = options.get_arguments()
#     encoder = Encoder()
#     decoder = Decoder(cfg.outViewN, cfg.outW, cfg.outH, cfg.renderDepth)
#     model = Structure_Generator()
