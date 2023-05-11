import pytorch_lightning as pl
import time
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os

class CustomModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = create_model(opt)
        self.visualizer = Visualizer(opt) # Web Visualizer

    def forward(self, x):
        # forward pass
        return self.model.set(x)

    def training_step(self, batch, batch_idx):
        data, _ = batch
        self.model.set_input(data)
        self.model.optimize_parameters()
        loss = self.model.loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.model.optimizer

    def train_dataloader(self):
        data_loader = CreateDataLoader(self.opt)
        return data_loader.load_data()