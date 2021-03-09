import pickle
from typing import List, Union

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from lofarnn.data.datasets import get_lotss_objects


class RadioSourceDataset(Dataset):
    """Radio Source dataset."""

    def __init__(
        self,
        json_file: Union[str, List[str]],
        vac_source: str,
        single_source_per_img: bool = True,
        num_sources: int = 40,
        shuffle: bool = False,
        transform=None,
        embed_source: bool = False,
        remove_no_source: bool = True,
        fraction: float = 1.0,
    ):
        """
        Args:
            json_file (string): Path to the json file with annotations
            single_source_per_img (bool, optional): Whether to give all sources with an image, or a single source per image
            embed_source (bool, optional): Whether to embed single sources in the images or not, instead of returning the
            radio image + source list, it then returns a single radio image with n channels, one for each available band
        """
        self.vac = get_lotss_objects(vac_source, verbose=False)
        if isinstance(json_file, str):
            self.annotations = pickle.load(open(json_file, "rb"), fix_imports=True)
        else:
            self.annotations = pickle.load(open(json_file[0], "rb"), fix_imports=True)
            for f in json_file[1:]:
                self.annotations.extend(pickle.load(open(f, "rb"), fix_imports=True))
        if fraction < 1.0:
            import random

            random.shuffle(self.annotations)
            self.annotations = self.annotations[: int(len(self.annotations) * fraction)]
        self.transform = transform
        self.embed_source = embed_source
        # Remove any non-standard files
        print(f"Len Anno: {len(self.annotations)}")
        print(f"JSON File: {json_file}")
        new_anno = []
        for anno in self.annotations:
            if isinstance(anno, np.ndarray):
                anno = anno.item()
            # print(np.count_nonzero(anno["optical_labels"][:num_sources]))
            #if anno["height"] == anno["width"] == 200:
            new_anno.append(anno)
            if (
                remove_no_source
                and np.count_nonzero(anno["optical_labels"][:num_sources]) != 0
            ):
                # Check if there is a source within the cutoff, if not, ignore it as well, unless we want non source ones
                new_anno.append(anno)
            elif not remove_no_source:
                new_anno.append(anno)
        self.annotations = new_anno
        print(f"Len Anno After Purge: {len(self.annotations)}")

        self.single_source = single_source_per_img
        self.num_sources = num_sources
        self.shuffle = shuffle

        # for length and indexing in if single optical source per image
        self.mapping = {}
        if self.single_source or self.embed_source:
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
        image = np.load(anno["file_name"], fix_imports=True)[:,:,0]
        image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        image = torch.from_numpy(image).float()
        radio_name = self._get_source_name(anno["file_name"])
        radio_source = self.vac[self.vac["Source_Name"] == radio_name]
        source = anno["optical_sources"][self.mapping[idx][1]]
        # New labels
        new_label = False
        if str(source[0]) == str(radio_source["objID"].data[0]) or (str(source[0]) == '999999' and str(radio_source["objID"].data[0]) == ''):
            if source[1] == radio_source["AllWISE"].data[0] or (str(source[1]) == 'N/A' and str(radio_source["AllWISE"].data[0]) == 'N/A'):
                new_label = True
        source = source[4:]  # Remove the IDs, etc.
        source[0] = source[0].value / (0.03)  # Distance (arcseconds)
        source[1] = source[1].value / (2 * np.pi)  # Convert to between 0 and 1
        source[2] = source[2] / 7.0  # Redshift
        source = np.asarray(source)
        #label = anno["optical_labels"][self.mapping[idx][1]]
        #if self.mapping[idx][1] == 0:
        #    label = True
        # First one is Optical, second one is Not
        if new_label:
            label = np.array([1, 0])  # True
        else:
            label = np.array([0, 1])  # False
        if self.transform:
            image, source = self.transform(image, source)
        return {
            "images": image,
            "sources": torch.from_numpy(source).float(),
            "labels": torch.from_numpy(label).float(),
            "names": self._get_source_name(anno["file_name"]),
        }

    def load_embedded_source(self, idx):
        anno = self.annotations[self.mapping[idx][0]]
        image = np.load(anno["file_name"].replace("/data/Research/", "/home/bieker/"), fix_imports=True)[:,:,0]
        radio_name = self._get_source_name(anno["file_name"])
        radio_source = self.vac[self.vac["Source_Name"] == radio_name]
        source = anno["optical_sources"][self.mapping[idx][1]]
        # New labels
        new_label = False
        if str(source[0]) == str(radio_source["objID"].data[0]) or (str(source[0]) == '999999' and str(radio_source["objID"].data[0]) == ''):
            if source[1] == radio_source["AllWISE"].data[0] or (str(source[1]) == 'N/A' and str(radio_source["AllWISE"].data[0]) == 'N/A'):
                new_label = True

        source = source[6:]  # Remove the IDs, Angles, etc. etc.
        #print(source)
        source[0] /= 7.0 # Redshift
        if source[0] < 0:
            source[0] = 0.0
        optical_channels = np.zeros(
            (image.shape[0], image.shape[1], len(source)), dtype=np.float
        )
        try:
            # Encode into the image one-hot encoding, appending n channels
            source_pix_loc = anno["optical_skycoords"][self.mapping[idx][1]].to_pixel(
                wcs=anno["wcs"]
            )
            for i, band in enumerate(source):
                optical_channels[int(source_pix_loc[0]), int(source_pix_loc[1]), i] = band
        except:
            new_label = False
        image = image.reshape((image.shape[0], image.shape[1], 1))
        image = np.concatenate((image, optical_channels), axis=2)
        image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image).float()
        #label = anno["optical_labels"][self.mapping[idx][1]]
        # First one is Optical, second one is Not
        if new_label:
            label = np.array([1, 0])  # True
        else:
            label = np.array([0, 1])  # False
        if self.transform:
            image, source = self.transform(image, source)
        return {
            "images": image,
            "labels": torch.from_numpy(label).float(),
            "sources": torch.from_numpy(
                np.zeros(
                    1,
                )
            ),
            "names": self._get_source_name(anno["file_name"]),
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
        image = np.load(
            anno["file_name"],
            fix_imports=True,
        )
        image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        image = image.reshape((1, anno["height"], anno["width"]))
        # print(image.shape)
        for i, item in enumerate(anno["optical_sources"]):
            anno["optical_sources"][i] = anno["optical_sources"][i][
                3:
            ]  # Remove the IDs, etc.
            anno["optical_sources"][i][0] = (
                anno["optical_sources"][i][0].value - 0.0
            ) / (
                0.03 - 0.0
            )  # Distance (arcseconds) for first 40 elements
            anno["optical_sources"][i][1] = anno["optical_sources"][i][1].value / (
                2 * np.pi
            )
            anno["optical_sources"][i][2] = (
                np.clip(anno["optical_sources"][i][2], 0.0, 7.0) - 0.0
            ) / (
                7.0 - 0.0
            )  # Redshift
        anno["optical_sources"].insert(
            0, [0 for _ in range(len(anno["optical_sources"][0]))]
        )
        if np.max(anno["optical_sources"][: (self.num_sources - 1)]) == 0:
            anno["optical_labels"].insert(
                0, 1
            )  # Give a 1 if no source in cutoff location out one for No Source
        else:
            anno["optical_labels"].insert(0, 0)
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
            sources = sources.reshape(1, sources.shape[0], sources.shape[1])
        image = torch.from_numpy(image).float()
        if self.transform:
            image, sources = self.transform(image, sources)
        return {
            "images": image,
            "sources": torch.from_numpy(sources).float(),
            "labels": torch.from_numpy(labels).float(),
            "names": self._get_source_name(anno["file_name"]),
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.single_source:
            return self.load_single_source(idx)
        elif self.embed_source:
            return self.load_embedded_source(idx)
        else:
            return self.load_multi_source(idx)


def collate_variable_fn(batch):
    images = []
    names = []
    max_size = 0
    for item in batch:
        names.append(item["names"])
        if (
            item["images"].shape[-1] > max_size
        ):  # last element should be either width or height, so good for this
            max_size = item["images"].shape[-1]

    if max_size > 400:
        max_size = 400

    # Second time to pad out tensors for this
    for item in batch:
        if item["images"].shape[-1] != max_size:
            images.append(
                torch.squeeze(
                    torch.nn.functional.interpolate(
                        item["images"].unsqueeze_(0), (max_size, max_size)
                    ),
                    0,
                )
            )
        else:
            images.append(item["images"])

    images = torch.stack(images, dim=0)
    data = torch.stack([item["sources"] for item in batch], dim=0)
    target = torch.stack([item["labels"] for item in batch], dim=0)
    return {"images": images, "sources": data, "labels": target, "names": names}


def collate_autokeras_fn(batch):
    images = []
    names = []
    max_size = 0
    for item in batch:
        names.append(item["names"])
        if (
            item["images"].shape[-1] > max_size
        ):  # last element should be either width or height, so good for this
            max_size = item["images"].shape[-1]

    if max_size > 400:
        max_size = 400

    # Second time to pad out tensors for this
    for item in batch:
        if item["images"].shape[-1] != max_size:
            images.append(
                torch.squeeze(
                    torch.nn.functional.interpolate(
                        item["images"].unsqueeze_(0), (max_size, max_size)
                    ),
                )
            )
        else:
            images.append(item["images"])

    images = torch.stack(images, dim=0)
    data = torch.stack([item["sources"] for item in batch], dim=0).squeeze_()
    target = torch.stack([item["labels"] for item in batch], dim=0)
    return {"images": images, "sources": data, "labels": target, "names": names}
