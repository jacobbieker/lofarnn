import pickle
from typing import Dict, Tuple, Any, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table


def get_lotss_objects(fname: str, verbose: bool = False) -> Table:
    """
    Load the LoTSS objects from a file
    """

    with fits.open(fname) as hdul:
        table = hdul[1].data
    print(table.columns)
    if verbose:
        print(table.columns)

    # convert from astropy.io.fits.fitsrec.FITS_rec to astropy.table.table.Table
    return Table(table)


def get_source_from_dict(source: Dict[str, Any]) -> str:
    if ".npy" in source["file_name"]:
        return source["file_name"].split("/")[-1].split(".npy")[0]
    else:
        return source["file_name"].split("/")[-1].split(".png")[0]


def get_lofar_dicts(annotation_filepath: str, fraction: float = 1.0) -> Dict[str, str]:
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    if fraction < 0.99999:
        # Only take subset of the dataset
        num_entries = len(dataset_dicts)
        num_kept = int(fraction * num_entries)
        step_size = int(num_entries / num_kept)
        new_dicts = []
        for i in range(0, len(dataset_dicts), step_size):
            new_dicts.append(dataset_dicts[i])
        dataset_dicts = new_dicts
    return dataset_dicts


def get_only_mutli_dicts(
    annotation_filepath: str, multi: bool = True, vac: str = ".."
) -> list:
    """
    Only get the multi or single component sources, whichever is wanted
    """
    vac_catalog = get_lotss_objects(vac)
    if multi:
        vac_catalog = vac_catalog[vac_catalog["LGZ_Assoc"] > 1]
    else:
        vac_catalog = vac_catalog[vac_catalog["LGZ_Assoc"] == 1]
    vac_catalog = vac_catalog["Source_Name"].data
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
        new_dicts = []
        for i in range(0, len(dataset_dicts)):
            name = dataset_dicts[i]["file_name"].split("/")[-1].split(".npy")[0]
            if (
                name in vac_catalog or name.rpartition(".")[0] in vac_catalog
            ):  # Needed for both rotated and non rotated
                new_dicts.append(dataset_dicts[i])
        dataset_dicts = new_dicts
    return dataset_dicts


def make_physical_dict(
    vac_catalog: str,
    size_cut: float = 0.0,
    flux_cut: float = 0.0,
    multi: bool = False,
    lgz: bool = True,
) -> Dict[str, str]:
    """
    Makes cuts in the value added catalog for the size, flux, and multi vs single component
    :param vac_catalog: Value-added catalog location
    :param size_cut: The angular size, in arcseconds, to make a cut into large and small ones
    :param flux_cut: The low range to cut into high brightness vs low brightness
    :param multi: Whether to split on multi component or not
    :return:
    """
    vac_catalog = get_lotss_objects(vac_catalog)
    physical_cut = {}
    if lgz:
        vac_catalog = vac_catalog[~np.isnan(vac_catalog["LGZ_Size"])]
    if multi:
        all_multi = vac_catalog[vac_catalog["LGZ_Assoc"] > 1]
        all_single = vac_catalog[vac_catalog["LGZ_Assoc"] == 1]
        physical_cut["multi_comp"] = all_multi["Source_Name"].data
        physical_cut["single_comp"] = all_single["Source_Name"].data
        physical_cut[f"size{size_cut}"] = vac_catalog[
            vac_catalog["LGZ_Size"] >= size_cut
        ]["Source_Name"].data
        physical_cut[f"flux{flux_cut}"] = vac_catalog[
            vac_catalog["Total_flux"] >= flux_cut
        ]["Source_Name"].data
        physical_cut[f"fluxLess{flux_cut}"] = vac_catalog[
            vac_catalog["Total_flux"] < flux_cut
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}"] = vac_catalog[
            vac_catalog["LGZ_Size"] < size_cut
        ]["Source_Name"].data
        physical_cut[f"size{size_cut}_flux{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] >= size_cut)
            & (vac_catalog["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}_flux{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] < size_cut)
            & (vac_catalog["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"size{size_cut}_fluxLess{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] >= size_cut)
            & (vac_catalog["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}_fluxLess{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] < size_cut)
            & (vac_catalog["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"multi_size{size_cut}_flux{flux_cut}"] = all_multi[
            (all_multi["LGZ_Size"] >= size_cut) & (all_multi["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"multi_sizeLess{size_cut}_flux{flux_cut}"] = all_multi[
            (all_multi["LGZ_Size"] < size_cut) & (all_multi["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"single_size{size_cut}_flux{flux_cut}"] = all_single[
            (all_single["LGZ_Size"] >= size_cut)
            & (all_single["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"single_sizeLess{size_cut}_flux{flux_cut}"] = all_single[
            (all_single["LGZ_Size"] < size_cut) & (all_single["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"multi_size{size_cut}_fluxLess{flux_cut}"] = all_multi[
            (all_multi["LGZ_Size"] >= size_cut) & (all_multi["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"multi_sizeLess{size_cut}_fluxLess{flux_cut}"] = all_multi[
            (all_multi["LGZ_Size"] < size_cut) & (all_multi["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"single_size{size_cut}_fluxLess{flux_cut}"] = all_single[
            (all_single["LGZ_Size"] >= size_cut) & (all_single["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"single_sizeLess{size_cut}_fluxLess{flux_cut}"] = all_single[
            (all_single["LGZ_Size"] < size_cut) & (all_single["Total_flux"] < flux_cut)
        ]["Source_Name"].data
    else:
        physical_cut[f"size{size_cut}"] = vac_catalog[
            vac_catalog["LGZ_Size"] >= size_cut
        ]["Source_Name"].data
        physical_cut[f"flux{flux_cut}"] = vac_catalog[
            vac_catalog["Total_flux"] >= flux_cut
        ]["Source_Name"].data
        physical_cut[f"fluxLess{flux_cut}"] = vac_catalog[
            vac_catalog["Total_flux"] < flux_cut
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}"] = vac_catalog[
            vac_catalog["LGZ_Size"] < size_cut
        ]["Source_Name"].data
        physical_cut[f"size{size_cut}_flux{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] >= size_cut)
            & (vac_catalog["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}_flux{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] < size_cut)
            & (vac_catalog["Total_flux"] >= flux_cut)
        ]["Source_Name"].data
        physical_cut[f"size{size_cut}_fluxLess{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] >= size_cut)
            & (vac_catalog["Total_flux"] < flux_cut)
        ]["Source_Name"].data
        physical_cut[f"sizeLess{size_cut}_fluxLess{flux_cut}"] = vac_catalog[
            (vac_catalog["LGZ_Size"] < size_cut)
            & (vac_catalog["Total_flux"] < flux_cut)
        ]["Source_Name"].data
    for key in physical_cut.keys():
        print(f"Cut: {key} | Num Elements: {len(physical_cut[key])}")
    return physical_cut


def get_physical_dicts(
    annotation_filepath: str, size: float = 0, vac_catalog: Optional[str] = None
) -> Tuple[list, list, list, list]:
    """
        Takes a subset of the annotation path based on criteria, such as multi component sources or image size, and
        returns four list[dict], split along multi- vs single component sources and size
    :param annotation_filepath:
    :param multi: Whether to take only multi component sources (True), only single component sources (False)
    :param size: Whether to make a cut on the size, in arcseconds, 0 no cut, keeps ones where the size is larger than the cut
    :param vac_catalog: LOFAR value added catalog
    :return: Multiple dicts of sources for Detectron2, with the applied cuts, useful to calculating the recall for various physical issues
    These dicts might be empty
    """
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    vac_catalog = get_lotss_objects(vac_catalog)
    multi_large = []
    multi_small = []
    single_large = []
    single_small = []
    for element in dataset_dicts:
        source_name = get_source_from_dict(element)
        vac_entry = vac_catalog[vac_catalog["Source_Name"] == source_name]
        if vac_entry["LGZ_Assoc"] > 1:
            if vac_entry["LGZ_Size"] >= size:
                multi_large.append(element)
            elif vac_entry["LGZ_Size"] < size:
                multi_small.append(element)
        elif vac_entry["LGZ_Assoc"] < 2:
            if vac_entry["LGZ_Size"] >= size:
                single_large.append(element)
            elif vac_entry["LGZ_Size"] < size:
                single_small.append(element)
    return multi_large, multi_small, single_large, single_small
