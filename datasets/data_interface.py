import inspect # 
import importlib # In order to dynamically import the library
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
#gcn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch_geometric
from torch_geometric.data import Batch
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset

#---->
def collate_MIL_survival_graph(batch):
    transposed = zip(*batch)
    return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples in transposed]


#########DTMIL
from typing import Optional, List
from torch import Tensor
import torch
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def collate_fn(batch):
    batch = list(zip(*batch))

    # in wsi, it is patch bag N * C * H * W, mask 1 * N
    batch[0] = torch.stack(batch[0][0])
    batch[1] = torch.stack(batch[1][0])

    tens = NestedTensor(batch[0], batch[1])
    return tens, default_collate([batch[2], batch[3], batch[4]])
##########

class DataInterface(pl.LightningDataModule):

    def __init__(self, seed=2021, train_batch_size=64, fold=0, train_num_workers=8, test_batch_size=1, test_num_workers=1, data_dir='', csv_path='', label_dir=''):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()
        self.fold = fold
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.label_dir = label_dir
        self.dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
                                               data_dir=data_dir,
                                               shuffle=False,
                                               seed=seed,
                                               print_info=True,
                                               patient_strat=False,
                                               n_bins=4,
                                               label_col='survival_months',
                                               ignore=[])


    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        train_dataset, val_dataset = self.dataset.return_splits(from_id=False, csv_path=self.label_dir+'splits_{}.csv'.format(self.fold))
        train_dataset.set_split_id(split_id=self.fold)
        val_dataset.set_split_id(split_id=self.fold)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset


    def train_dataloader(self):
        weights = make_weights_for_balanced_classes_split(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=WeightedRandomSampler(weights, len(weights)),num_workers=self.train_num_workers,  collate_fn=collate_MIL_survival_graph)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size,  sampler=SequentialSampler(self.val_dataset), num_workers=self.train_num_workers,  collate_fn=collate_MIL_survival_graph)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,  sampler=SequentialSampler(self.test_dataset), num_workers=self.train_num_workers,  collate_fn=collate_MIL_survival_graph)



def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)