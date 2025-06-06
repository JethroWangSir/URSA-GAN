from argparse import ArgumentParser
from typing import Any, Union
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import os
from pytorch_lightning.loggers import WandbLogger

from dataset import TAT, TAT_DM
from module.feature import Mel_Spectrogram
import score as score
from loss import softmax, amsoftmax
    
class Task(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 10,
        max_epochs: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mel_trans = Mel_Spectrogram()
        
        from module.resnet import resnet34, resnet18, resnet34_large
        from module.ecapa_tdnn import ecapa_tdnn, ecapa_tdnn_large
        from module.transformer_cat import transformer_cat
        from module.conformer import conformer
        from module.conformer_cat import conformer_cat
        from module.conformer_weight import conformer_weight

        if self.hparams.encoder_name == "resnet18":
            self.encoder = resnet18(embedding_dim=self.hparams.embedding_dim)
            print('Initialized resnet18')
        elif self.hparams.encoder_name == "resnet34":
            self.encoder = resnet34_large(embedding_dim=self.hparams.embedding_dim)
            print('Initialized resnet34')
        elif self.hparams.encoder_name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim)
            print('Initialized ecapa_tdnn')
        elif self.hparams.encoder_name == "ecapa_tdnn_large":
            self.encoder = ecapa_tdnn_large(embedding_dim=self.hparams.embedding_dim)
            print('Initialized ecapa_tdnn_large')
        elif self.hparams.encoder_name == "conformer":
            self.encoder = conformer(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)
            print('Initialized conformer')
        elif self.hparams.encoder_name == "transformer_cat":
            self.encoder = transformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)
            print('Initialized transformer_cat')
        elif self.hparams.encoder_name == "conformer_cat":
            self.encoder = conformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer,
                    pos_enc_layer_type=self.hparams.pos_enc_layer_type)
            print('Initialized conformer_cat')
        elif self.hparams.encoder_name == "conformer_weight":
            self.encoder = conformer_weight(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)
            print('Initialized conformer_weight')
        else:
            raise ValueError("encoder name error")
        
        if self.hparams.loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)
        else:
            self.loss_fun = softmax(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)

    def forward(self, x):
        feature = self.mel_trans(x)
        embedding = self.encoder(feature)
        return embedding

    def training_step(self, batch, batch_idx):
        print('Start training...')

        waveform, label = batch

        waveform = waveform.squeeze(1)

        feature = self.mel_trans(waveform)

        embedding = self.encoder(feature)

        loss, acc = self.loss_fun(embedding, label)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # print('Start validating...')
        waveform, label = batch
        waveform = waveform.squeeze(1)
        with torch.no_grad():
            feature = self.mel_trans(waveform)
            self.encoder.eval()
            embedding = self.encoder(feature)
        loss, acc = self.loss_fun(embedding, label)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer,
                    optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--num_workers", default=10, type=int)
        parser.add_argument("--embedding_dim", default=256, type=int)
        parser.add_argument("--num_classes", type=int, default=6)
        parser.add_argument("--max_epochs", type=int, default=30)
        parser.add_argument("--num_blocks", type=int, default=6)
        parser.add_argument("--input_layer", type=str, default="conv2d")
        parser.add_argument("--pos_enc_layer_type", type=str, default="rel_pos")
        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=80)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=2000)
        parser.add_argument("--weight_decay", type=float, default=0.000001)
        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--loss_name", type=str, default="amsoftmax")
        parser.add_argument("--encoder_name", type=str, default="resnet34")
        parser.add_argument("--train_csv_path", type=str, default="data/train.csv")
        parser.add_argument("--trial_path", type=str, default="data/vox1_test.txt")
        parser.add_argument("--score_save_path", type=str, default=None)
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--aug', action='store_true')
        return parser

def cli_main():
    parser = ArgumentParser()
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Task(**vars(args))

    dm = TAT_DM(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=3,
        mode='max',
        filename='{epoch:02d}-{val_acc:.2f}',
        dirpath=args.save_dir
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        sync_batchnorm=True,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.save_dir,
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=int(200 / args.batch_size),
        log_every_n_steps=25,
        precision='16-mixed'
    )
    
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    cli_main()
