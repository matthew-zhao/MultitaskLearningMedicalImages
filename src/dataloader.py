import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from replacement_random_sampler import ReplacementRandomSampler
from dataset import CustomDataset, ImageDataset

class TrainViewDataLoader(DataLoader):
    def __iter__(self):
        batch = torch.Tensor()
        batch2 = torch.Tensor()
        for idx in self.sampler:
            print("In for loop")
            data, labels = self.dataset[idx]
            data_batch = torch.cat([batch, data])
            label_batch = torch.cat([batch2, labels])
            while data_batch.size(0) >= self.batch_size:
                print("in while loop")
                if data_batch.size(0) == self.batch_size:
                    yield [data_batch, label_batch]
                    batch = torch.Tensor()
                    batch2 = torch.Tensor()
                else:
                    return_data_batch, data_batch = data_batch.split([self.batch_size,data_batch.size(0)-self.batch_size])
                    return_label_batch, label_batch = label_batch.split([self.batch_size,label_batch.size(0)-self.batch_size])
                    yield [return_data_batch, return_label_batch]
        if data_batch.size(0) > 0 and not self.drop_last:
            print("in last if check")
            yield data_batch, label_batch


class TestViewDataLoader(DataLoader):
    def __iter__(self):
        for idx in self.sampler:
            batch = torch.Tensor()
            batch2 = torch.Tensor()
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

class MURALoader(BaseDataLoader):
    def __init__(self, data_task_list, batch_size=128, num_minibatches=5, train=True, shuffle=True, drop_last=False, rescale_size=224,
            sample_with_replacement=True):
        super(MURALoader, self).__init__(batch_size, train, shuffle, drop_last)
        if train:
            data_transform = transforms.Compose([
                transforms.Resize((rescale_size, rescale_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((rescale_size, rescale_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

        self.phase = 'train' if train else 'valid'
        
        image_datasets = [ImageDataset(data[self.phase], transform=data_transform) for data in data_task_list]
        samplers = None
        if sample_with_replacement:
            samplers = [ReplacementRandomSampler(image_dataset, num_minibatches * batch_size) for image_dataset in image_datasets]
            if self.phase == 'train':
                self.dataloaders = [TrainViewDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=drop_last,
                                     sampler=sampler) for dataset, sampler in zip(image_datasets, samplers)]
            else:
                self.dataloaders = [TestViewDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=drop_last,
                                     sampler=sampler) for dataset, sampler in zip(image_datasets, samplers)]
        else:
            if self.phase == 'train':
                self.dataloaders = [TrainViewDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=drop_last) for dataset in image_datasets]
            else:
                self.dataloaders = [TestViewDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=drop_last) for dataset in image_datasets]
        # replace with None if this doesn't work
        self.task_dataloader = self.dataloaders

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler_list = samplers if samplers else None


    def _create_TaskDataLoaders(self):
        self.task_dataloader = []
        sampler_idx = 0
        for dataloader in self.dataloaders:
            sampler = self.sampler_list[sampler_idx]
            images = []
            labels = []
            for batch_images, batch_labels in dataloader:
                for i in batch_images:
                    images.append(i)
                for l in batch_labels:
                    labels.append(l)

            dataset = CustomDataset(data=images.copy(), labels=labels.copy())
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    drop_last=self.drop_last,
                                    sampler=sampler)
            self.task_dataloader.append(dataloader)
            sampler_idx += 1


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

    def num_classes_multi(self, num_tasks):
        return [2 for _ in range(num_tasks)]

class MultiTaskDataLoader:
    def __init__(self, dataloaders, phase, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

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
        if self.step >= self.size:
            self.step = 0
            print("StopIter raised because of step and size")
            raise StopIteration

        # Uncomment below if we want to choose a random task per batch
        #self.task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        # if self.phase != 'train' or self.views_remaining == 0:
        try:
            data, labels = self.iters[self.task].__next__()
        except StopIteration:
            # Uncomment below if we want to choose a random task per batch
            # self.iters[self.task] = iter(self.dataloaders[self.task])
            if self.task + 1 >= len(self.iters):
                print("StopIter raised because of task greater than iters")
                raise StopIteration
            self.task += 1
            data, labels = self.iters[self.task].__next__()
        # self.views_remaining = list(data.size())[0]
        self.step += 1
        # self.views_remaining -= 1
        # if self.phase == 'train':
        #     # separate each view in image
        #     data = data[self.views_remaining]
        return data, labels, self.task