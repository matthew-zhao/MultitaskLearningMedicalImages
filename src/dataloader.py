import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from replacement_random_sampler import ReplacementRandomSampler
from dataset import CustomDataset, ImageDataset

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
    def __init__(self, data_task_list, batch_size=128, num_minibatches=5, train=True, shuffle=False, drop_last=False, rescale_size=224):
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

        phase = 'train' if train else 'valid'
        
        image_datasets = [ImageDataset(data[phase], transform=data_transform) for data in data_task_list]
        samplers = [ReplacementRandomSampler(image_dataset, num_minibatches * batch_size) for image_dataset in image_datasets]
        self.dataloaders = [DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=drop_last,
                                     sampler=sampler) for dataset, sampler in zip(image_datasets, samplers)]
        # replace with None if this doesn't work
        self.task_dataloader = self.dataloaders

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler_list = samplers


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
        return MultiTaskDataLoader(self.task_dataloader, prob)

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
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task