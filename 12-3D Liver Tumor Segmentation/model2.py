import torch

class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)

    
    
class UNet(torch.nn.Module):
    """
    This class implements a UNet for segmentation
    with reduced parameters (approximately half).
    """
    def __init__(self):
        """Sets up the U-Net Structure with reduced parameters."""
        super().__init__()
        
        ############# DOWN #####################
        self.layer1 = DoubleConv(1, 16)  # Reduced from 32 to 16
        self.layer2 = DoubleConv(16, 32)  # Reduced from 64 to 32
        self.layer3 = DoubleConv(32, 64)  # Reduced from 128 to 64
        self.layer4 = DoubleConv(64, 128)  # Reduced from 256 to 128

        #########################################

        ############## UP #######################
        self.layer5 = DoubleConv(128 + 64, 64)  # Adjusted inputs
        self.layer6 = DoubleConv(64 + 32, 32)  # Adjusted inputs
        self.layer7 = DoubleConv(32 + 16, 16)  # Adjusted inputs
        self.layer8 = torch.nn.Conv3d(16, 3, 1)  # Output unchanged (3 classes)
        #########################################

        self.maxpool = torch.nn.MaxPool3d(2)

    def forward(self, x):
        
        ####### DownConv 1#########
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        ###########################
        
        ####### DownConv 2#########        
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        ###########################

        ####### DownConv 3#########        
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        ###########################
        
        ##### Intermediate Layer ## 
        x4 = self.layer4(x3m)
        ###########################

        ####### UpCONV 1#########        
        x5 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  # Upsample with a factor of 2
        x5 = torch.cat([x5, x3], dim=1)  # Skip-Connection
        x5 = self.layer5(x5)
        ###########################

        ####### UpCONV 2#########        
        x6 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5)        
        x6 = torch.cat([x6, x2], dim=1)  # Skip-Connection    
        x6 = self.layer6(x6)
        ###########################
        
        ####### UpCONV 3#########        
        x7 = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)       
        x7 = self.layer7(x7)
        ###########################
        
        ####### Predicted segmentation#########        
        ret = self.layer8(x7)
        return ret
