#! /usr/bin/env python

import os
from argparse import ArgumentParser
from functools import partial
import h5py
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import torch
from PIL import ImageFilter, Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18, BlinkEstimationModelResnet50, \
    BlinkEstimationModelVGG16, BlinkEstimationModelVGG19, BlinkEstimationModelDenseNet121
from rtbene_dataset import RTBENEH5Dataset
from torchmetrics import AveragePrecision


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha: float, gamma: float):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self._alpha, self._gamma, reduction="mean")


class TrainRTBENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects, class_weights=None):
        super(TrainRTBENE, self).__init__()
        if hparams.loss_fn == "bce":
            assert class_weights is not None, "Class Weights can't be None if using Binary Cross Entropy as the loss function"

        _loss_fn = {
            "bce": partial(torch.nn.BCELoss, pos_weight=torch.Tensor([class_weights[1]])),
            "fl": partial(FocalLoss, alpha=0.25, gamma=2)
        }

        _models = {
            "resnet18": BlinkEstimationModelResnet18,
            "resnet50": BlinkEstimationModelResnet50,
            "vgg16": BlinkEstimationModelVGG16,
            "vgg19": BlinkEstimationModelVGG19,
            "densenet121": BlinkEstimationModelDenseNet121
        }
        self._model = _models.get(hparams.model_base)()
        self._criterion = _loss_fn.get(hparams.loss_fn)()
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self.save_hyperparameters(hparams)

        # for adaptive data augmentation
        self._theta = 0
        self._data_train = None

    def forward(self, left_patch, right_patch):
        return self._model(left_patch, right_patch)

    def training_step(self, batch, batch_idx):
        _left, _right, _label = batch
        _pred_blink = self.forward(_left, _right)
        loss = self._criterion(_pred_blink, _label)

        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _left, _right, _label = batch
        _pred_blink = self.forward(_left, _right)
        loss = self._criterion(_pred_blink, _label)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        _left, _right, _label = batch
        _pred_blink = self.forward(_left, _right)
        return {'test_label': _label, 'test_prediction': _pred_blink}

    def validation_epoch_end(self, outputs):
        _losses = torch.stack([x['val_loss'] for x in outputs])
        val_loss = _losses.mean()

        # check the loss against the training loss and change the probability of the transforms.
        if self._data_train is not None and self.hparams.curriculum is True:
            self._theta = self._theta + 1 / self.hparams.max_epochs
            self._theta = np.clip(self._theta, 0, 1)
            _transforms = transforms.Compose([transforms.RandomResizedCrop(size=(36, 60), scale=(1 - self._theta * 0.5, 1 + self._theta * 0.5), interpolation=Image.BICUBIC),
                                              transforms.RandomPerspective(distortion_scale=self._theta * 0.4, p=1.0, interpolation=Image.BICUBIC),
                                              transforms.RandomRotation(degrees=10, expand=False),
                                              transforms.ColorJitter(brightness=self._theta * 0.5, hue=self._theta * 0.5, contrast=self._theta * 0.5, saturation=self._theta * 0.5),
                                              lambda x: x if np.random.random_sample() >= self._theta else x.filter(ImageFilter.GaussianBlur(radius=1)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
            self._data_train.transform = _transforms
        elif self._data_train is not None and self.hparams.curriculum is False:
            _transforms = transforms.Compose([transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3), interpolation=Image.BICUBIC),
                                              transforms.RandomPerspective(distortion_scale=0.2, interpolation=Image.BICUBIC),
                                              transforms.RandomRotation(degrees=10, expand=False),
                                              transforms.RandomGrayscale(p=0.1),
                                              transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5,
                                                                     saturation=0.5),
                                              lambda x: x if np.random.random_sample() <= 0.1 else x.filter(
                                                  ImageFilter.GaussianBlur(radius=3)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
            self._data_train.transform = _transforms

        self.log("val_loss", val_loss)

    def test_epoch_end(self, outputs):
        _labels = torch.stack([x['test_label'] for x in outputs])
        _predictions = torch.stack([x['test_prediction'] for x in outputs])

        average_precision = AveragePrecision(pos_label=1)
        mAP = average_precision(_predictions, _labels)

        self.log("mAP", mAP)

    def configure_optimizers(self):
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        _optimizer = torch.optim.AdamW(_params_to_update, lr=self.hparams.learning_rate)

        return _optimizer

    def train_dataloader(self):
        self._data_train = RTBENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"),
                                           subject_list=self._train_subjects,
                                           loader_desc="train")
        return DataLoader(self._data_train, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        _data_validate = RTBENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"),
                                         subject_list=self._validate_subjects, loader_desc="valid")
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_io_workers, pin_memory=True)

    def test_dataloader(self):
        _data_validate = RTBENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"),
                                         subject_list=self._test_subjects, loader_desc="valid")
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_io_workers, pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--loss_fn', choices=["bce", "fl"], default="bce")
        parser.add_argument('--batch_size', default=1024, type=int)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--model_base', choices=["vgg16", "vgg19", "resnet18", "resnet50", "densenet121"],  default="vgg16")
        parser.add_argument('--curriculum', action="store_true", dest="curriculum")
        parser.set_defaults(curriculum=False)
        parser.set_defaults(augment=False)
        return parser


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))

    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--gpu', type=int, default=1,
                              help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2', action="append")
    _root_parser.add_argument('--hdf5_file', type=str,
                              default=os.path.abspath("/home/ahmed/datasets/rtbene_dataset.hdf5"))
    _root_parser.add_argument('--dataset', type=str, choices=["rt_bene"], default="rt_bene")
    _root_parser.add_argument('--save_dir', type=str, default=os.path.abspath(
        os.path.join(root_dir, '../../rt_bene_model_training/pytorch/checkpoints')))
    _root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
    _root_parser.add_argument('--no-benchmark', action='store_false', dest="benchmark")
    _root_parser.add_argument('--num_io_workers', default=6, type=int)
    _root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    _root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    _root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    _root_parser.add_argument('--seed', type=int, default=0)
    _root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    _root_parser.add_argument('--max_epochs', type=int, default=40,
                              help="Maximum number of epochs to perform; the trainer will Exit after.")
    _root_parser.set_defaults(benchmark=True)
    _root_parser.set_defaults(k_fold_validation=True)

    _model_parser = TrainRTBENE.add_model_specific_args(_root_parser, root_dir)
    _hyperparams = _model_parser.parse_args()

    pl.seed_everything(_hyperparams.seed)

    _train_subjects = []
    _valid_subjects = []
    _test_subjects = []
    if _hyperparams.dataset == "rt_bene":
        if _hyperparams.k_fold_validation:
            # 6 is discarded
            _train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
            _train_subjects.append([3, 4, 7, 9, 5, 12, 13, 14])
            _train_subjects.append([5, 12, 13, 14, 1, 2, 8, 10])

            _valid_subjects.append([5, 12, 13, 14])
            _valid_subjects.append([1, 2, 8, 10])
            _valid_subjects.append([3, 4, 7, 9])

            _test_subjects.append([0, 11, 15, 16])
            _test_subjects.append([0, 11, 15, 16])
            _test_subjects.append([0, 11, 15, 16])
        else:  # we want to train with the entire dataset
            print('Training on the whole dataset - do not use the trained model for evaluation purposes!')
            print('Validation dataset is a subject included in training...use at your own peril!')

            # 6 is discarded as per the paper
            _train_subjects.append([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            _valid_subjects.append([7])
            _test_subjects.append([7])
    else:
        raise NotImplementedError("No other dataset is currently implemented")

    for fold, (train_s, valid_s, _test_s) in enumerate(zip(_train_subjects, _valid_subjects, _test_subjects)):
        # this is a hack to get class weights, i'm sure there's a better way fo doing it but I can't think of it
        with h5py.File(_hyperparams.hdf5_file, mode="r") as _h5_f:
            _class_weights = RTBENEH5Dataset.get_class_weights(h5_file=_h5_f, subject_list=train_s)

        _model = TrainRTBENE(hparams=_hyperparams,
                             train_subjects=train_s,
                             validate_subjects=valid_s,
                             test_subjects=_test_s,
                             class_weights=_class_weights)

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename=f'fold={fold}-' + '{epoch}-{val_loss:.3f}', save_top_k=10)
        _annealing_epochs = int(_hyperparams.max_epochs - (0.8 * _hyperparams.max_epochs))
        swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, annealing_epochs=_annealing_epochs, annealing_strategy="cos")

        # start training
        trainer = Trainer(gpus=_hyperparams.gpu,
                          callbacks=[checkpoint_callback, swa_callback],
                          precision=32,
                          progress_bar_refresh_rate=1,
                          log_every_n_steps=5,
                          flush_logs_every_n_steps=10,
                          min_epochs=_hyperparams.min_epochs,
                          max_epochs=_hyperparams.max_epochs,
                          accumulate_grad_batches=_hyperparams.accumulate_grad_batches,
                          benchmark=_hyperparams.benchmark)
        trainer.fit(_model)
