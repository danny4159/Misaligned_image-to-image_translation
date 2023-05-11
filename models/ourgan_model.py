import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from pytorch_msssim import ssim
from torch import nn
from torchvision import models

class ourGAN(BaseModel):
    def name(self):
        return 'ourGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.vgg=VGG16().cuda()
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self.netG(self.real_A)
        self.real_B = self.input_B
    

    def test(self):
        # no backprop gradients
        with torch.no_grad():
            self.real_A = self.input_A #[1,3,256,256]
            self.fake_B = self.netG(self.real_A) #[1,1,256,256]
            self.real_B = self.input_B #[1,1,256,256]

        # PSNR 계산
        mse = nn.MSELoss()(self.fake_B, self.real_B)
        psnr = 10 * torch.log10(1 / mse)

        # SSIM 계산
        ssim_value = ssim(self.fake_B, self.real_B, data_range=2.0, size_average=True) # data_range: -1~1
        
        # TODO: LPIPS 계산 추가 (https://github.com/cszhilu1998/RAW-to-sRGB)
        # https://github.com/cszhilu1998/RAW-to-sRGB/blob/master/metrics.py => RGB에 맞게 LPIPS 수정 필요
        
        return psnr.item(), ssim_value.item()

        ##############FID Score를 위한 코드. 실패###############
        # fake_B와 real_B간의 FID를 구해 각 slice 별로
        # Compute FID score between fake_B and real_B
        
        # # InceptionV3 model을 불러옵니다.
        # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        # inception_model = InceptionV3([block_idx]).cuda()

        # # fake_B와 real_B를 InceptionV3에 적합한 크기와 정규화로 조정합니다.
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

        # with torch.no_grad():
        #     fake_B_resized = F.interpolate(self.fake_B.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')
        #     real_B_resized = F.interpolate(self.real_B.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')

        #     fake_B_processed = normalize(fake_B_resized[0]).unsqueeze(0)
        #     real_B_processed = normalize(real_B_resized[0]).unsqueeze(0)
            
        #     # InceptionV3 모델을 사용해 fake_B와 real_B의 특성을 추출합니다.
        #     fake_B_features = inception_model(fake_B_processed.cuda())[0]
        #     real_B_features = inception_model(real_B_processed.cuda())[0]

        #     # fake_B와 real_B의 특성에 대한 평균 및 공분산 행렬을 계산합니다.
        #     mu_fake, sigma_fake = calculate_activation_statistics_from_features(fake_B_features.cpu().numpy().reshape(fake_B_features.size(0), -1))
        #     mu_real, sigma_real = calculate_activation_statistics_from_features(real_B_features.cpu().numpy().reshape(real_B_features.size(0), -1))

        #     # FID score를 계산합니다.
        #     fid = fid_score.calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)

        # return fid
    
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # real_A: [batch,3,256,256] , fake_B: [batch,1,256,256]
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data) # condition을 이렇게 넣어줬구나. 가짜라고 판별해야하는 것에 정답도 은근슬쩍 넣어서 Discriminator가 
        pred_fake = self.netD(fake_AB.detach()) # pred_fake: [batch,1,256,256] , fake_AB: [batch,4,256,256]
        self.loss_D_fake = self.criterionGAN(pred_fake, False) # pred_fake shape로 0(False)를 채워줌. 그리고 loss를 구한다

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        #Perceptual loss
        self.VGG_real=self.vgg(self.real_B.expand( [ int(self.real_B.size()[0]), 3, int(self.real_B.size()[2]), int(self.real_B.size()[3]) ] ))[0] # real_B: [16,1,256,256] , real_B.expand [16,3,256,256]
        self.VGG_fake=self.vgg(self.fake_B.expand( [ int(self.real_B.size()[0]), 3, int(self.real_B.size()[2]), int(self.real_B.size()[3]) ] ))[0] # VGG: [16,128,128,128] , VGG[0]: [128,128,128]
        self.VGG_loss=self.criterionL1(self.VGG_fake,self.VGG_real)* self.opt.lambda_vgg
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.VGG_loss
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_VGG', self.VGG_loss.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data[:,0,:,:].unsqueeze(1)) # 코드에 차원 안맞는 에러가 있어서 수정해줌
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        # 모두 [1,1,256,256]
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

# def calculate_activation_statistics_from_features(features, eps=1e-6):
#     mu = np.mean(features, axis=0)
#     sigma = np.cov(features, rowvar=False)
#     return mu, sigma 

#Extracting VGG feature maps before the 2nd maxpooling layer  
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2
    
    
""" model.py를 이해하기 위한 코드 필요
ex)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    model = ourGAN(opt)
    
    
"""