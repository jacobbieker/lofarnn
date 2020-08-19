from torch.utils.data import Dataset
import torch
import json
import pickle
import numpy as np


class RadioSourceDataset(Dataset):
    """Radio Source dataset."""

    def __init__(self, json_file, single_source_per_img=True, num_sources=40, shuffle=False):
        """
        Args:
            json_file (string): Path to the json file with annotations
            single_source_per_img (bool, optional): Whether to give all sources with an image, or a single source per image
        """
        self.annotations = pickle.load(open(json_file, "rb"), fix_imports=True)
        self.single_source = single_source_per_img
        self.num_sources = num_sources
        self.shuffle = shuffle

        # for length and indexing in if single optical source per image
        self.mapping = {}
        if self.single_source:
            total = 0
            for i, annotation in enumerate(self.annotations):
                for j, _ in enumerate(annotation["optical_sources"]):
                    self.mapping[total] = (i, j)
                    total += 1
            self.length = total
        else:
            self.length = len(self.annotations)

    def __len__(self):
        return self.length

    def load_single_source(self, idx):
        """
        Given a single index, get the single source, image, and label for it
        """
        anno = self.annotations[self.mapping[idx][0]]
        image = np.load(anno["image"], fix_imports=True)
        source = anno["optical_sources"][self.mapping[idx][1]]
        label = anno["optical_labels"][self.mapping[idx][1]]

        return {
            "image": torch.from_numpy(image),
            "sources": torch.from_numpy(source),
            "labels": torch.from_numpy(label),
        }

    def load_multi_source(self, idx):
        """
        Given single index, get all the sources and labels, shuffling the order
        """
        anno = self.annotations[idx]
        image = np.load(anno["image"], fix_imports=True)
        sources = np.asarray(anno["optical_sources"])
        labels = np.asarray(anno["optical_labels"])

        # Shuffling order of sources and labels, and only taking x number of sources
        sources = sources[: self.num_sources]
        labels = labels[: self.num_sources]
        if self.shuffle:
            indices = np.indices(labels.shape)
            np.random.shuffle(indices)
            sources = sources[indices]
            labels = labels[indices]

        return {
            "image": torch.from_numpy(image),
            "sources": torch.from_numpy(sources),
            "labels": torch.from_numpy(labels),
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.single_source:
            return self.load_single_source(idx)
        else:
            return self.load_multi_source(idx)
