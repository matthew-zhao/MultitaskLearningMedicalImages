import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from albumentations import (
    Compose, OneOf, HorizontalFlip, ShiftScaleRotate, JpegCompression, Blur, CLAHE, RandomGamma, RandomContrast, RandomBrightness, Resize, PadIfNeeded
)
from replacement_random_sampler import ReplacementRandomSampler
from dataset import ImageDataset, MURADataset, ChexpertDataset

class TrainViewDataLoader(DataLoader):
    def __iter__(self):
        data_batch = torch.Tensor()
        label_batch = torch.Tensor().long()
        for idx in self.sampler:
            data, labels = self.dataset[idx]
            data_batch = torch.cat([data_batch, data])
            label_batch = torch.cat([label_batch, labels])
            while data_batch.size(0) >= self.batch_size:
                if data_batch.size(0) == self.batch_size:
                    yield [data_batch, label_batch]
                    data_batch = torch.Tensor()
                    label_batch = torch.Tensor().long()
                else:
                    return_data_batch, data_batch = data_batch.split([self.batch_size,data_batch.size(0)-self.batch_size])
                    return_label_batch, label_batch = label_batch.split([self.batch_size,label_batch.size(0)-self.batch_size])
                    yield [return_data_batch, return_label_batch]
        if data_batch.size(0) > 0 and not self.drop_last:
            #print("in last if check")
            yield data_batch, label_batch


class TestViewDataLoader(DataLoader):
    def __iter__(self):
        for idx in self.sampler:
            batch = torch.Tensor()
            batch2 = torch.Tensor().long()
            data, labels = self.dataset[idx]
            index = torch.tensor([0])
            yield torch.cat([batch, data]), torch.index_select(torch.cat([batch2, labels]), 0, index)

class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError

class RadiographLoader(BaseDataLoader):
    def __init__(self, data_task_list, batch_size=128, num_minibatches=5, train=True, shuffle=True, drop_last=False, rescale_size=224,
            sample_with_replacement=True):
        super(RadiographLoader, self).__init__(batch_size, train, shuffle, drop_last)
        self.phase = 'train' if train else 'valid'
        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if train:
            data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale_size, rescale_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomApply([
                    transforms.RandomAffine(10, translate=(0.0625, 0.0625), scale=(0.85, 1.15)),
                ], p=0.5),
                #transforms.RandomResizedCrop(rescale_size),
                #transforms.RandomRotation(30)
            ])

            albumentations_transforms = Compose([
                OneOf([
                    JpegCompression(quality_lower=80),
                    Blur(),
                ], p=0.5),
                OneOf([
                    CLAHE(),
                    RandomGamma(),
                    RandomContrast(),
                    RandomBrightness(),
                ], p=0.5)
            ], p=0.5)

            second_data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale_size, rescale_size))
            ])

            albumentations_transforms = None

            second_data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        image_datasets = {}
        for study_type, data in data_task_list:
            if study_type != 'chexpert':
                image_datasets[study_type] = MURADataset(data[self.phase], transform=data_transform, second_transform=second_data_transform, albumentations_transforms=albumentations_transforms)
            else:
                image_datasets[study_type] = ChexpertDataset(data[self.phase], transform=data_transform, second_transform=second_data_transform, albumentations_transforms=albumentations_transforms)

        samplers = None
        if sample_with_replacement:
            samplers = {study_type: ReplacementRandomSampler(image_dataset) for study_type, image_dataset in image_datasets.items()}
            self._create_TaskDataLoaders(image_datasets, samplers=samplers)
        else:
            self._create_TaskDataLoaders(image_datasets)

        self.sampler_list = samplers if samplers else None

    def _create_TaskDataLoaders(self, image_datasets, samplers=None):
        self.dataloaders = {}
        for study_type, dataset in image_datasets.items():
            if self.phase == 'train' and study_type != 'chexpert':
                self.dataloaders[study_type] = TrainViewDataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False if samplers else True,
                                                 drop_last=self.drop_last,
                                                 sampler=samplers[study_type] if samplers else None)
            elif self.phase == 'valid' and study_type != 'chexpert':
                self.dataloaders[study_type] = TestViewDataLoader(dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False if samplers else self.shuffle,
                                                drop_last=self.drop_last,
                                                sampler=samplers[study_type] if samplers else None)
            else:
                self.dataloaders[study_type] = DataLoader(dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False if samplers else self.shuffle,
                                                drop_last=self.drop_last,
                                                sampler=samplers[study_type] if samplers else None)
        # replace with None if this doesn't work
        self.task_dataloader = self.dataloaders

    def get_loader(self, prob='uniform'):
        if self.task_dataloader is None:
            self._create_TaskDataLoaders()
        return MultiTaskDataLoader(self.task_dataloader, self.phase, prob)

    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(10))
        else:
            assert task in list(range(10)), 'Unknown task: {}'.format(task)
            labels = [0 for _ in range(10)]
            labels[task] = 1
            return labels


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 3

    def num_classes_multi(self, uncertainty_strategy):
        num_classes_dict = {}
        for study_type, task_dataloader in self.task_dataloader.items():
            if study_type != 'chexpert':
                num_classes_dict[study_type] = 2
            elif uncertainty_strategy == 'u_multiclass':
                num_classes_dict[study_type] = 18
            else:
                num_classes_dict[study_type] = 6
        return num_classes_dict

class MultiTaskDataLoader:
    '''
    dataloaders is a dictionary mapping study_type to dataloader
    '''
    def __init__(self, dataloaders, phase, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [(study_type, iter(dataloader)) for study_type, dataloader in self.dataloaders.items()]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0

        self.task = 0
        # self.phase = phase

        # if this is set to true, we don't call __next__ on iters
        # self.views_remaining = 0
        # self.data_label_dict = None


    def __iter__(self):
        return self


    def __next__(self):
        #if self.step >= self.size:
        #    self.step = 0
            #print("StopIter raised because of step and size")
        #    raise StopIteration

        # Uncomment below if we want to choose a random task per batch
        #self.task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        # if self.phase != 'train' or self.views_remaining == 0:
        try:
            study_type, loader_iter = self.iters[self.task]
            data, labels, level = loader_iter.__next__()
        except StopIteration:
            # Uncomment below if we want to choose a random task per batch
            # self.iters[self.task] = iter(self.dataloaders[self.task])
            if self.task + 1 >= len(self.iters):
                #print("StopIter raised because of task greater than iters")
                raise StopIteration
            self.task += 1
            study_type, loader_iter = self.iters[self.task]
            data, labels, level = loader_iter.__next__()
        # self.views_remaining = list(data.size())[0]
        self.step += 1
        # self.views_remaining -= 1
        # if self.phase == 'train':
        #     # separate each view in image
        #     data = data[self.views_remaining]
        return data, labels, level, study_type