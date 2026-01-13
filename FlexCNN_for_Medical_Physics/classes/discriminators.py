from torch import nn

#################################
#### SINOGRAMS DISCRIMINATOR ####
#################################

class Disc_S_90(nn.Module):
    '''
    Through experimentation it has been found that sinogram discriminators work best with a fat network neck.
    This class takes as input a 90x90.
    '''
    def __init__(self, config):
        super(Disc_S_90, self).__init__()

        input_channels = config['gen_sino_channels']
        hidden_dim = config['IS_disc_hidden_dim']
        patchGAN = config['IS_disc_patchGAN']

        ## Sequence 1 ##
        self.seq1 = nn.Sequential(
            # Sinogram Shape: (in_channels,90,90)
            # nn.Conv2d: Hf = [Hi+2*padding-dilation(kernel-1)-1]/stride + 1
            #               = [Hi+2*padding-kernel]/stride + 1 (for dialation=1)

            # Feature Map Block
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=7, padding=3, padding_mode='reflect'),

            # Contracting Block without normalization:
            # H1 = (90-4)/2+1 = 44
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=4, stride=2, padding=0, padding_mode='reflect'),
                nn.LeakyReLU(negative_slope=0.2),

            # Contracting Blocks:
            # H1 = (44-4)/2+1 = 21
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*3, kernel_size=4, stride=2, padding=0, padding_mode='reflect'),
                nn.InstanceNorm2d(hidden_dim*3), nn.LeakyReLU(negative_slope=0.2),
            # H1 = (21-4)/2+1 = 9.5 = 9
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*4, kernel_size=4, stride=2, padding=0, padding_mode='reflect'),
                nn.InstanceNorm2d(hidden_dim*4), nn.LeakyReLU(negative_slope=0.2),
            # H1 = (9-4)/2+1 = 3.5 = 3
            nn.Conv2d(in_channels=hidden_dim*4, out_channels=hidden_dim*5, kernel_size=4, stride=2, padding=0, padding_mode='reflect'),
                nn.InstanceNorm2d(hidden_dim*5), nn.LeakyReLU(negative_slope=0.2),
        )

        ## PatchGAN ##
        if patchGAN==True:
            # H = [3+2*1-3]/1+1 = 3 (3x3x3 matrix)
            self.seq2 = nn.Sequential(
                nn.Conv2d(hidden_dim*5, hidden_dim*5, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm2d(hidden_dim*5), nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.seq2 = nn.Sequential(
                # H0 = (3-3)/1+1 = 1 (1x1x3 matrix)
                nn.Conv2d(in_channels=hidden_dim*5, out_channels=hidden_dim*5, kernel_size=3),
                    nn.BatchNorm2d(hidden_dim*5), nn.LeakyReLU(negative_slope=0.2),
            )
        ## 1x1 Convolution ##
        self.seq3 = nn.Conv2d(hidden_dim * 5, input_channels, kernel_size=1)

    def forward(self, image):
        a = self.seq1(image)
        b = self.seq2(a) # a tensor
        c = self.seq3(b)
        #return disc_pred.view(len(disc_pred), -1) # returns a flattened tensor
        return c.squeeze()

##############################
#### IMAGES DISCRIMINATOR ####
##############################

class Disc_I_90(nn.Module):
    '''
    Image discriminator. Takes 90x90 images as input.
    Automatically derives input_channels from config['gen_image_channels'].
    '''
    def __init__(self, config):
        super(Disc_I_90, self).__init__()

        input_channels = config['gen_image_channels']
        hidden_dim = config['SI_disc_hidden_dim']
        patchGAN = config['SI_disc_patchGAN']

        ## Sequence 1 ##
        self.seq1 = nn.Sequential(
            # Image Shape: (1,90,90)
            # nn.Conv2d: Hf = [Hi+2*padding-dilation(kernel-1)-1]/stride + 1
            #               = [Hi+2*padding-kernel]/stride + 1 (for dialation=1)

            # H = [90-4]/2+1 = 44
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=4, stride=2),
                nn.BatchNorm2d(hidden_dim), nn.LeakyReLU(negative_slope=0.2),
            # H = [44-4]/2+1 = 21
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=4, stride=2),
                nn.BatchNorm2d(hidden_dim*2), nn.LeakyReLU(negative_slope=0.2),
            # H = [21-4]/2+1 = 9.5 = 9
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*4, kernel_size=4, stride=2),
                nn.BatchNorm2d(hidden_dim*4), nn.LeakyReLU(negative_slope=0.2),
            # H = [9+2-4]/2+1 = 4.5 = 4
            nn.Conv2d(in_channels=hidden_dim*4, out_channels=hidden_dim*4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim*4), nn.LeakyReLU(negative_slope=0.2),
        )

        ## Sequence 2 ##
        if patchGAN==True:
            # H = [4+2-3]/1+1 = 4
            self.seq2=nn.Conv2d(hidden_dim*4, 1, kernel_size=3, padding=1, padding_mode='reflect')
        else:
            # H = [4-4]/2+1 = 1
            self.seq2=nn.Conv2d(hidden_dim*4, 1, kernel_size=4, stride=2)

    def forward(self, image):

        a = self.seq1(image)
        disc_pred = self.seq2(a) # a tensor
        #return disc_pred.view(len(disc_pred), -1) # returns a flattened tensor
        return disc_pred.squeeze()