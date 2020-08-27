from torch.utils.data import Dataset
import torch
import json
import pickle
import numpy as np
import astropy.units as u


class RadioSourceDataset(Dataset):
    """Radio Source dataset."""

    def __init__(self, json_file, single_source_per_img=True, num_sources=40, shuffle=False, norm=True):
        """
        Args:
            json_file (string): Path to the json file with annotations
            single_source_per_img (bool, optional): Whether to give all sources with an image, or a single source per image
        """
        self.annotations = pickle.load(open(json_file, "rb"), fix_imports=True)
        self.norm = norm
        # Remove any non-standard files
        print(f"Len Anno: {len(self.annotations)}")
        new_anno = []
        for anno in self.annotations:
            if isinstance(anno, np.ndarray):
                anno = anno.item()
            #if anno["height"] == anno["width"] == 200:
            new_anno.append(anno)
        self.annotations = new_anno
        print(f"Len Anno After Purge: {len(self.annotations)}")

        self.single_source = single_source_per_img
        self.num_sources = num_sources
        self.shuffle = shuffle

        # for length and indexing in if single optical source per image
        self.mapping = {}
        if self.single_source:
            total = 0
            for i, annotation in enumerate(self.annotations):
                for j, _ in enumerate(annotation["optical_sources"]):
                    if j < self.num_sources:
                        self.mapping[total] = (i, j)
                        total += 1
            self.length = total
        else:
            self.length = len(self.annotations)
        print(f"Total Items: {self.length}")

    def __len__(self):
        return self.length

    def load_single_source(self, idx):
        """
        Given a single index, get the single source, image, and label for it
        """
        anno = self.annotations[self.mapping[idx][0]]
        image = np.load(anno["file_name"], fix_imports=True)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        source = anno["optical_sources"][self.mapping[idx][1]]
        source[0] = source[0].value
        source[1] = source[1].value / (2*np.pi) # Convert to between 0 and 1
        source = np.asarray(source)
        label = anno["optical_labels"][self.mapping[idx][1]]
        # First one is Optical, second one is Not
        if label:
            label = np.array([1,0])
        else:
            label = np.array([0,1])
        return {
            "image": torch.from_numpy(image).float(),
            "sources": torch.from_numpy(source).float(),
            "labels": torch.from_numpy(label).float(),
        }

    @staticmethod
    def _get_source_name(name):
        source_name = name.split("/")[-1].split(".cnn")[0]
        return source_name

    def load_multi_source(self, idx):
        """
        Given single index, get all the sources and labels, shuffling the order
        """
        anno = self.annotations[idx]
        image = np.load(anno["file_name"], fix_imports=True)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        #print(image.shape)
        for i, item in enumerate(anno["optical_sources"]):
            anno["optical_sources"][i][0] = anno["optical_sources"][i][0].value
            anno["optical_sources"][i][1] = anno["optical_sources"][i][1].value / (2*np.pi)
            if self.norm:
                for j in range(2,len(anno["optical_sources"][i])):
                    value = anno["optical_sources"][i][j]
                    value = np.clip(value, 10.0, 28.0)
                    anno["optical_sources"][i][j] = (value - 10.0) / (28.0 - 10.0)
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
            labels = labels.reshape(self.num_sources)
        else:
            sources = sources.reshape(1,sources.shape[0], sources.shape[1])
        return {
            "image": torch.from_numpy(image).float(),
            "sources": torch.from_numpy(sources).float(),
            "labels": torch.from_numpy(labels).float(),
            "source_names": self._get_source_name(anno["file_name"])
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.single_source:
            return self.load_single_source(idx)
        else:
            return self.load_multi_source(idx)
