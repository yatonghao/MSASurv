from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset



class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        #new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        

        self.cls_ids_prep()


        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        r"""
        
        """
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""
        
        """

        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata,   data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split#, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None




class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir,  **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

        save_path = os.path.join(data_dir, 'ctrans_20x/pth_files_4_4')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(data_dir, 'ctrans_10x/pth_files_4_4')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def preprocess_features(self, npdata, pca=-1):

        _, ndim = npdata.shape
        assert npdata.dtype == np.float32

        if np.any(np.isnan(npdata)):
            raise Exception('nan occurs')

        if pca != -1:
            import faiss
            print('\nPCA form dim {} to dim {}'.format(ndim, pca))
            mat = faiss.PCAMatrix(ndim, pca, eign_power=-0.5)
            mat.train(npdata)
            assert mat.is_trained
            npdata = mat.apply_py(npdata)

        if np.any(np.isnan(npdata)):
            percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
            if percent > 0.1:
                raise Exception('more than 0.1% nan occurs after pca, percent: {}%'.format(percent))
            else:
                npdata[np.isnan(npdata)] = 0.0

        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

        return npdata

    def cluster_dataset(self, features, k_cluster: int = 8, method='kmeans'):
        prototype_list = []
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k_cluster, random_state=1337, n_init=10)
            kmeans.fit_predict(features)
            prototype_list.append(kmeans.cluster_centers_)
            labels = kmeans.labels_
            grouped_features = [[] for _ in range(k_cluster)]
            for i in range(len(labels)):
                label = labels[i]
                grouped_features[label].append(features[i])
            for i in range(len(grouped_features)):
                grouped_features[i] = torch.from_numpy(np.array(grouped_features[i]))

            return grouped_features


    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        save_path_20x = os.path.join(data_dir, 'ctrans_20x/pth_files_4_4', '{}.pth'.format(case_id))
        if not os.path.exists(save_path_20x):
            path_features_20x = []
            for slide_id in slide_ids:
                wsi_path = os.path.join(data_dir, 'ctrans_20x/pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                try:
                    path_features_20x.append(torch.load(wsi_path))
                except:
                    pass
            path_features_20x = torch.cat(path_features_20x, dim=0).numpy()
            path_features_20x = self.preprocess_features(path_features_20x, pca=-1)
            path_features_20x = self.cluster_dataset(features=path_features_20x, k_cluster=4, method='kmeans')
            torch.save(path_features_20x, save_path_20x)
        else:
            path_features_20x = torch.load(save_path_20x)

        save_path_10x = os.path.join(data_dir, 'ctrans_10x/pth_files_4_4', '{}.pth'.format(case_id))
        if not os.path.exists(save_path_10x):
            path_features_10x = []
            for slide_id in slide_ids:
                wsi_path_10x = os.path.join(data_dir, 'ctrans_10x/pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                try:
                    path_features_10x.append(torch.load(wsi_path_10x))
                except:
                    pass
            path_features_10x = torch.cat(path_features_10x, dim=0).numpy()
            path_features_10x = self.preprocess_features(path_features_10x, pca=-1)
            path_features_10x = self.cluster_dataset(features=path_features_10x, k_cluster=4, method='kmeans')
            torch.save(path_features_10x, save_path_10x)
        else:
            path_features_10x = torch.load(save_path_10x)

        return path_features_20x, path_features_10x, label, event_time, c




        # path_features = []
        # for slide_id in slide_ids:
        #     wsi_path = os.path.join(data_dir, 'ctrans_20x/pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
        #     try:
        #         path_features.append(torch.load(wsi_path))
        #     except:
        #         pass
        # path_features = torch.cat(path_features, dim=0)
        # return path_features,  label, event_time, c


        # path_features = []
        # from datasets.BatchWSI import BatchWSI
        # for slide_id in slide_ids:
        #     wsi_path = os.path.join(data_dir, 'ctrans_20x/graph_files', '{}.pt'.format(slide_id.rstrip('.svs')))
        #     try:
        #         wsi_bag = torch.load(wsi_path)
        #         path_features.append(wsi_bag)
        #     except:
        #         pass
        # path_features = BatchWSI.from_data_list(path_features, update_cat_dims={'edge_latent': 1})
        # return path_features, label, event_time, c


        # path_features = []
        # for slide_id in slide_ids:
        #     wsi_path_0 = os.path.join(data_dir,  '{}_0.pt'.format(slide_id.rstrip('.svs')))
        #     wsi_path_1 = os.path.join(data_dir, '{}_1.pt'.format(slide_id.rstrip('.svs')))
        #     try:
        #         wsi_bag_0 = torch.load(wsi_path_0)
        #         wsi_bag_1 = torch.load(wsi_path_1)
        #         path_features.append(wsi_bag_0)
        #         path_features.append(wsi_bag_1)
        #     except:
        #         pass
        #
        # return path_features, label, event_time, c




class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata,
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))


    def __len__(self):
        return len(self.slide_data)

    ### <--
    def set_split_id(self, split_id):
        self.split_id = split_id


