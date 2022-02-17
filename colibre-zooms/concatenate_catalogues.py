"""
Concatenates halo catalogues into a single file
"""

import h5py as h5
import numpy as np
from typing import List
from glob import glob
import argparse


def load_catalogues(path: str, filename_patterns: List[str]) -> List[str]:
    """
    Finds halo catalogues.

    Parameters
    ----------
    path: str
    Path to the directory with halo catalogues

    filename_patterns: List[str]
    Filename patters to use when searching for halo catalogues

    Returns
    -------
    output: List[str]
    List with the paths to halo catalogues
    """

    catalogue_names = []
    for filename_pattern in filename_patterns:
        paths_to_files = f"{path}/{filename_pattern}"
        catalogue_names += glob(paths_to_files)

    N_zooms = len(catalogue_names)

    print(f"Total number of zooms found: {N_zooms} \n")
    for count, catalogue_name in enumerate(catalogue_names, start=1):
        print(f"{count}. {catalogue_name}")

    print(" ")
    return catalogue_names


def concatenate_catalogues(paths: List[str], filename: str):
    """
    Performs concatenation. Saves the concatenated data into a new file
    Parameters
    ----------
    paths: str
    Path to halo catalogues to be concatenated.

    filename: str
    Name of the output file

    Returns
    -------
    """

    N_zooms = len(paths)

    if N_zooms > 0:
        concatenated_data = h5.File(filename, "w")
    else:
        raise IOError("No data were provided for concatenation!!!")

    # Keep record which files are concatenated
    concatenated_data.create_group("ConcatenatedFiles")

    for count, path in enumerate(paths, start=1):
        print(f"\n Opening zoom catalogue {count}...")
        with h5.File(path, "r") as data:

            # Append 'path' to keep record
            concatenated_data["ConcatenatedFiles"].attrs.create(path, "")

            # Copy global attributes
            if count == 1:
                for glb_attr in list(data.attrs):
                    concatenated_data.attrs.create(glb_attr, data.attrs[glb_attr])

            # Loop over fields
            for key in list(data.keys()):
                # Create new fields based on the data from the first processed catalogue
                if count == 1:
                    try:
                        # Chunks = True and max shape = None enable resize method
                        concatenated_data.create_dataset(
                            key,
                            data=data[key][:],
                            shape=data[key].shape,
                            chunks=True,
                            dtype=data[key].dtype,
                            maxshape=(None,),
                        )

                    # Groups like 'SimulationInfo' have only attributes and no values
                    except AttributeError:
                        print(
                            f"Unable to fetch data for key '{key}'. Creating group... "
                        )
                        concatenated_data.create_group(key)

                    # Copy attributes for each field
                    for attr in list(data[key].attrs):
                        concatenated_data[key].attrs.create(attr, data[key].attrs[attr])

                # The field has already been created. Use resize to append new data
                else:
                    try:
                        # Fetch fields
                        concatenated_field = concatenated_data[key]
                        new_field = data[key]

                        # Compute new shape
                        new_shape = (concatenated_field.shape[0] + new_field.shape[0],)

                        # Append new data via resize method
                        concatenated_field.resize(new_shape)
                        concatenated_field[-new_field.shape[0] :] = new_field[:]

                    # Fields like 'SimulationInfo' have only attributes and no values
                    except AttributeError:
                        print(
                            f"Unable to fetch data for key '{key}'. Skipping "
                            f"resize..."
                        )

    concatenated_data.close()

    print("The concatenated data have been generated!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Description:
        Concatenates halo catalogues from zoom-in simulations into a new file."""
    )

    parser.add_argument(
        "-p",
        "--pattern",
        nargs="+",
        required=True,
        help="Filename pattern to match when searching for input halo catalogues. E.g. "
        "'halo_halo_*_00??.properties*'",
    )

    parser.add_argument(
        "-i",
        "--input-path",
        nargs="?",
        help="Path to halo catalogues that will be concatenated into a single file. "
        "Default: './'",
        default="./",
    )

    parser.add_argument(
        "-o",
        "--output-name",
        nargs="?",
        help="Name of the output file. Default: './out.properties'",
        default="./out.properties",
    )

    config = parser.parse_args()

    print("-------------------------------------")
    print(f"1. Path to the directory with input files: {config.input_path}")
    print("2. Patterns used for searching of input catalogues")
    for pattern in config.pattern:
        print(f"\t {pattern}")
    print(f"3.Name of output file: {config.output_name}")
    print("------------------------------------- \n")

    paths = load_catalogues(path=config.input_path, filename_patterns=config.pattern)
    concatenate_catalogues(paths=paths, filename=config.output_name)
