import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

cifar100_finelabels = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']

cifar100_coarselabels = ['aquatic_mammals','fish','flowers','food_containers','fruit_and_vegetables','household_electrical_devices','household_furniture','insects','large_carnivores','large_man-made_outdoor_things','large_natural_outdoor_scenes','large_omnivores_and_herbivores','medium_mammals','non-insect_invertebrates','people','reptiles','small_mammals','trees','vehicles_1','vehicles_2']

class Cifar100(nn.Module):
    path = Path("data/cifar100/CIFAR-100")
    files_raw = ["meta", "train", "test" ]
    files = ["images.npy", "labels.npy", "coarse_labels.npy"]

    def __init__(self, test=False, coarse=False):
        super(Cifar100, self).__init__()
        self.files_raw = [self.path / f for f in self.files_raw]
        self.files = [self.path / f for f in self.files]

        if all([f.exists() for f in self.files]):
            # load pre-processed data
            self.images = np.load(self.files[0], allow_pickle=True)
            self.labels = np.load(self.files[2 if coarse else 1], allow_pickle=True)
        else:
            # pre-process data and save
            self._preProcess(coarse)

        self.images = torch.from_numpy(self.images.astype(np.float32) / 255)
        if test:
            self.images, self.labels = self.images[-10_000:], self.labels[-10_000:]
        else:
            self.images, self.labels = self.images[:-10_000], self.labels[:-10_000]

        self.batch_size = len(self.labels)
        self.num_batches = 1

    def _preProcess(self, coarse=False):
        ''' Saves the raw data in numpy format'''

        data = [self._loadFromFile(f) for f in self.files_raw[1:]]
        self.images = np.r_[*[i for i, _, _ in data]]
        self.labels = np.r_[*[l for _, l, _ in data]]
        self.coarse_labels = np.r_[*[l for _, _, l in data]]
        np.save(self.files[0], self.images)
        np.save(self.files[1], self.labels)
        np.save(self.files[2], self.coarse_labels)
        if coarse: self.labels = self.coarse_labels

    def _loadFromFile(self, filename):
        ''' Returns (images[_, 3, 32, 32], labels)'''
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data[b'data'].reshape(-1, 3, 32, 32), data[b'fine_labels'], data[b'coarse_labels']
    

