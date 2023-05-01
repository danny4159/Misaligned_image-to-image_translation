import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
import torchio as tio

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    #data directory
    target_file=opt.dataroot+'/'+opt.phase+'/data.mat'
    f = h5py.File(target_file,'r')              # return data_x, data_y
    print("data_x shape: ", f['data_x'].shape)
    print("data_y shape: ", f['data_y'].shape)
    slices=np.array(f['data_x']).shape[3]//2    # size(data_x,data_y) = 256,256,2275,3  (x,y,slice,slice한칸 옮긴 결과)
    samples=range(np.array(f['data_y']).shape[2])
    #Selecting neighbouring slices based on the inputs
    if opt.which_direction=='AtoB':
        data_x=np.array(f['data_x'])[:,:,:,slices-opt.input_nc//2:slices+opt.input_nc//2+1]
        data_y=np.array(f['data_y'])[:,:,:,slices-opt.output_nc//2:slices+opt.output_nc//2+1]
    else:            
        data_y=np.array(f['data_y'])[:,:,:,slices-opt.input_nc//2:slices+opt.input_nc//2+1]
        data_x=np.array(f['data_x'])[:,:,:,slices-opt.output_nc//2:slices+opt.output_nc//2+1]
    #Shuffle slices in data_y for the cGAN case (incase the input data is registered)
    if opt.dataset_mode == 'unaligned_mat':  
        if opt.isTrain:
            print("Training phase")
            random.shuffle(samples)
        else:
            print("Testing phase")
        data_y=data_y[:,:,samples,:]
    #데이터에 misaling 적용할지 말지 (입력하면 True, 입력안하면 False) ex. --dataset_misalign
    if opt.dataset_misalign == True:
        print("▶ Dataset Misalign Appling...")
        print("opt.data_misalign: ", opt.dataset_misalign)
        data_y = np.transpose(data_y,(3,0,1,2)) # 3,256,256,2275
        seed = 0 if opt.training else 10000  # 환자별로 transform 고정. train은 0부터 test는 10000부터
        for i in range(0, data_y.shape[3], 91): # 0에서 2274까지 91 간격으로 (환자마다 91개 slice데이터가 있고, 각각 transform을 다르게 적용하기위해)
            print("seed: ",seed)
            torch.manual_seed(seed)
            seed += 1
            data_y_patient = data_y[:,:,:,i:i+91]
            # transform = tio.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 1, 1), degrees=(0,0,10))
            # transform = tio.RandomAffine(scales=(0.99, 1.01, 0.99, 1.01, 1, 1), degrees=(0,0,1))
            transform = tio.RandomAffine(scales=(0.95, 1.05, 0.95, 1.05, 1, 1), degrees=(0,0,5))
            data_y[:,:,:,i:i+91] = transform(data_y_patient)
        data_y = np.transpose(data_y,(1,2,3,0)) # 256,256,2275,3 (원위치)
        print("---------------- Misalign complete ----------------")
    data_x=np.transpose(data_x,(3,2,0,1)) # 3,2275,256,256
    data_y=np.transpose(data_y,(3,2,0,1))
    #Ensure that there is no value less than 0
    data_x[data_x<0]=0  
    data_y[data_y<0]=0 
    dataset=[]
    #making range of each image -1 to 1 and converting to torch tensor (각 slice 마다 norm)
    for train_sample in range(data_x.shape[1]):
        data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
        data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
        dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 
        'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
    #If number of samples in data_x and data_y are different (for unregistered images) make them same    
    if data_x.shape[1]!=data_y.shape[1]:    
        for train_sample in range(max(data_x.shape[1],data_y.shape[1])):
            if data_x.shape[1]>=data_y.shape[1] and train_sample>(data_y.shape[1]-1):
                data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
                dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'A_paths':opt.dataroot})  
            elif data_y.shape[1]>data_x.shape[1] and train_sample>(data_x.shape[1]-1):  
                data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
                dataset.append({ 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 'B_paths':opt.dataroot})  
            else:
                data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
                data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
                dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 
                'A_paths':opt.dataroot, 'B_paths':opt.dataroot})               

    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self


    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data