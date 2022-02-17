"""
Concatenates snapshots into a single file
"""

import h5py as h5
import numpy as np
from typing import List, Dict
from glob import glob
import argparse


fields_gas = [
    "AtomicHydrogenMasses",
    "Densities",
    "DensitiesAtLastAGNEvent",
    "DensitiesAtLastSupernovaEvent",
    "DustMassFractions",
    "DustMassesFromTable",
    "ElementMassFractions",
    "Entropies",
    "GraphiteMasses",
    "HeliumMasses",
    "HydrogenMasses",
    "InternalEnergies",
    "IonisedHydrogenMasses",
    "LastAGNFeedbackScaleFactors",
    "LastKineticEarlyFeedbackScaleFactors",
    "LastSNIIKineticFeedbackScaleFactors",
    "LastSNIIKineticFeedbackvkick",
    "LastSNIIThermalFeedbackScaleFactors",
    "LastSNIaThermalFeedbackScaleFactors",
    "Masses",
    "MaximalSNIIKineticFeedbackvkick",
    "MaximalTemperatureScaleFactors",
    "MaximalTemperatures",
    "MetalMassFractions",
    "Pressures",
    "SilicatesMasses",
    "SpeciesFractions",
    "StarFormationRates",
    "SubgridPhysicalDensities",
    "SubgridTemperatures",
    "Temperatures",
    "SmoothingLengths",
]

fields_stars = [
    "BirthDensities",
    "BirthScaleFactors",
    "BirthTemperatures",
    "BirthVelocityDispersions",
    "DensitiesAtLastAGNEvent",
    "DensitiesAtLastSupernovaEvent",
    "ElementMassFractions",
    "EnergiesReceivedFromAGNFeedback",
    "FeedbackEnergyFractions",
    "InitialMasses",
    "LastAGNFeedbackScaleFactors",
    "LastKineticEarlyFeedbackScaleFactors",
    "LastSNIIKineticFeedbackScaleFactors",
    "LastSNIIKineticFeedbackvkick",
    "LastSNIIThermalFeedbackScaleFactors",
    "LastSNIaThermalFeedbackScaleFactors",
    "MetalMassFractions",
    "Masses",
    "MaximalSNIIKineticFeedbackvkick",
    "MaximalTemperatureScaleFactors",
    "MaximalTemperatures",
    "SubgridBirthDensities",
    "SubgridBirthTemperatures",
    "SmoothingLengths",
]

fields_bhs = [
    "AGNTotalInjectedEnergies",
    "AccretedAngularMomenta",
    "AccretionLimitedTimeSteps",
    "AccretionRates",
    "CumulativeNumberOfSeeds",
    "DynamicalMasses",
    "ElementMasses",
    "EnergyReservoirs",
    "FormationScaleFactors",
    "GasCircularVelocities",
    "GasCurlVelocities",
    "GasDensities",
    "GasRelativeVelocities",
    "GasSoundSpeeds",
    "GasVelocityDispersions",
    "LastAGNFeedbackScaleFactors",
    "LastHighEddingtonFractionScaleFactors",
    "LastMajorMergerScaleFactors",
    "LastMinorMergerScaleFactors",
    "MetalMasses",
    "NumberOfAGNEvents",
    "NumberOfDirectSwallows",
    "NumberOfGasNeighbours",
    "NumberOfHeatingEvents",
    "NumberOfMergers",
    "NumberOfRepositionAttempts",
    "NumberOfRepositions",
    "NumberOfSwallows",
    "SubgridMasses",
    "SwallowedAngularMomenta",
    "TotalAccretedMasses",
    "SmoothingLengths",
]


def load_snapshots(path: str, filename_patterns: List[str]) -> List[str]:
    """
    Finds snapshots.

    Parameters
    ----------
    path: str
    Path to the directory with snapshots

    filename_patterns: List[str]
    Filename patters to use when searching for snapshots

    Returns
    -------
    output: List[str]
    List with the paths to snapshots
    """

    snapshot_names = []
    for filename_pattern in filename_patterns:
        paths_to_files = f"{path}/{filename_pattern}"
        snapshot_names += glob(paths_to_files)

    N_zooms = len(snapshot_names)

    print(f"Total number of zooms found: {N_zooms} \n")
    for count, snapshot_name in enumerate(snapshot_names, start=1):
        print(f"{count}. {snapshot_name}")

    print(" ")
    return snapshot_names


def concatenate_snapshots(
    paths: List[str], filename: str, groups: Dict[str, List[str]], name: bytes
):
    """
    Performs concatenation. Saves the concatenated data into a new file
    Parameters
    ----------
    paths: str
    Path to snapshots to be concatenated.

    filename: str
    Name of the output file

    groups: Dict[str,List[str]]
    Names of groups containing fields to do concatenation for

    name: bytes
    Name to put in the header of the output file

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

    # Loop over input files
    for count, path in enumerate(paths, start=1):
        print(f"\n Opening zoom snapshot {count}...")
        with h5.File(path, "r") as data:

            # Keep track of what has been processed
            concatenated_data["ConcatenatedFiles"].attrs.create(
                path, data["Parameters"].attrs["MetaData:run_name"]
            )

            # Copy global attributes
            if count == 1:
                for glb_attr in list(data.attrs):
                    concatenated_data.attrs.create(glb_attr, data.attrs[glb_attr])

            # Loop over fields
            for group in list(data.keys()):
                print(f"Processing {group}...")

                # Create new groups based on the data from the first processed snapshot
                if group in groups:

                    if count == 1:

                        # Create groups
                        concatenated_data.create_group(group)

                        # Copy attributes
                        for attr in list(data[group].attrs):
                            concatenated_data[group].attrs.create(
                                attr, data[group].attrs[attr]
                            )

                        for field in groups[group]:
                            try:
                                # Compute maxshape based on shape
                                if len(data[group][field].shape) == 1:
                                    maxshape = (None,)
                                elif len(data[group][field].shape) == 2:
                                    maxshape = (None, data[group][field].shape[1])
                                else:
                                    raise IndexError(
                                        f"Unexpected shape in group {group} in field "
                                        f"{field} '{data[group][field].shape}'!!!"
                                    )

                                # Chunks = True and max shape = None enable resize
                                # method
                                concatenated_data[group].create_dataset(
                                    field,
                                    data=data[group][field][:],
                                    shape=data[group][field].shape,
                                    chunks=True,
                                    dtype=data[group][field].dtype,
                                    maxshape=maxshape,
                                )
                            except AttributeError:
                                print(
                                    f"Field {field} in group {group} is a group!!! "
                                    f"Creating group..."
                                )
                                concatenated_data[group].create_group(field)

                                # Unlike other fields, field 'NamedColumns' has
                                # subfields and does not have attributes. It thus needs
                                # special care...
                                if field == "NamedColumns":
                                    print("Processing 'NamedColumns'...")

                                    for sub_field in data[group][field].keys():
                                        concatenated_data[group][field].create_dataset(
                                            sub_field,
                                            data=data[group][field][sub_field][:],
                                            shape=data[group][field][sub_field].shape,
                                            dtype=data[group][field][sub_field].dtype,
                                        )

                            # Copy attributes for each field in the group
                            for attr in list(data[group][field].attrs):
                                concatenated_data[group][field].attrs.create(
                                    attr, data[group][field].attrs[attr]
                                )

                    # The field has already been created. Use resize to append new data
                    else:
                        # Header deserves special care
                        if group == "Header":
                            print("Updating particle numbers in the header...")

                            # We want to update the total number of particles saved
                            # in the header (for each particle type)
                            for attr in ["NumPart_ThisFile", "NumPart_Total"]:
                                concatenated_data[group].attrs[attr] = [
                                    x + y
                                    for x, y in zip(
                                        concatenated_data[group].attrs[attr],
                                        data[group].attrs[attr],
                                    )
                                ]

                        for field in groups[group]:
                            try:
                                # Fetch fields
                                concatenated_field = concatenated_data[group][field]
                                new_field = data[group][field]

                                # Compute new shape
                                if len(concatenated_field.shape) == 1:
                                    new_shape = (
                                        concatenated_field.shape[0]
                                        + new_field.shape[0],
                                    )
                                    # Append new data via resize method
                                    concatenated_field.resize(new_shape)
                                    concatenated_field[
                                        -new_field.shape[0] :
                                    ] = new_field[:]

                                elif len(concatenated_field.shape) == 2:

                                    # Sanity check
                                    assert (
                                        concatenated_field.shape[1]
                                        == new_field.shape[1]
                                    )

                                    new_shape = (
                                        concatenated_field.shape[0]
                                        + new_field.shape[0],
                                        concatenated_field.shape[1],
                                    )

                                    # Append new data via resize method
                                    concatenated_field.resize(new_shape)
                                    concatenated_field[
                                        -new_field.shape[0] :, :
                                    ] = new_field[:, :]

                                else:
                                    raise IndexError(
                                        f"Unexpected shape in group {group} in field "
                                        f"{field} '{concatenated_field.shape}'!!!"
                                    )
                            except AttributeError:
                                print(
                                    f"Field {field} in group {group} is a group!!!. "
                                    f"Skipping resize..."
                                )

    # Record how many zooms were concatenated. Also change header name to 'name'
    if "Header" in groups:
        concatenated_data["Header"].attrs.create("NumberOfZooms", len(paths))
        try:
            concatenated_data["Header"].attrs["RunName"] = name
        except AttributeError:
            print("Can't find attribute 'RunName' in 'Header'. Creating" "one...")
            concatenated_data["Header"].attrs.create("RunName", name)

    if "Parameters" in groups:
        try:
            concatenated_data["Parameters"].attrs["MetaData:run_name"] = name
        except AttributeError:
            print(
                "Can't find attribute 'MetaData:run_name' in 'Parameters'. Creating"
                "one..."
            )
            concatenated_data["Parameters"].attrs.create("MetaData:run_name", name)

    concatenated_data.close()

    print("The concatenated data have been generated!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Description:
        Concatenates snapshots from zoom-in simulations into a new file."""
    )

    parser.add_argument(
        "-p",
        "--pattern",
        nargs="+",
        required=True,
        help="Filename pattern to match when searching for snapshots from different "
        "zooms. E.g. 'shapshot_halo_*_00??.hdf5'",
    )

    parser.add_argument(
        "-i",
        "--input-path",
        nargs="?",
        help="Path to snapshots that will be concatenated into a single file. "
        "Default: './'",
        default="./",
    )

    parser.add_argument(
        "-o",
        "--output-name",
        nargs="?",
        help="Name of the output file. Default: './out.hdf5'",
        default="./out.hdf5",
    )

    parser.add_argument(
        "-n",
        "--header-name",
        nargs="?",
        help="Name in the header of the concatenated snapshot. Default: "
        "'Concatenated_data'",
        default="Concatenated_data",
    )

    config = parser.parse_args()

    print("-------------------------------------")
    print(f"1. Path to the directory with input files: {config.input_path}")
    print("2. Patterns used for searching of snapshots from zoom simulations")
    for pattern in config.pattern:
        print(f"\t {pattern}")
    print(f"3.Name of output file: {config.output_name}")
    print("------------------------------------- \n")

    paths = load_snapshots(path=config.input_path, filename_patterns=config.pattern)

    # Relevant groups to look for for concatenation. We don't want to concatenate data
    # that we are not going to plot. That is why below there are so many empty lists
    relevant_groups = {
        "PartType0": fields_gas,
        "PartType1": [],
        "PartType2": [],
        "PartType4": fields_stars,
        "PartType5": fields_bhs,
        "Code": [],
        "Cosmology": [],
        "Units": [],
        "UnusedParameters": [],
        "HydroScheme": [],
        "InternalCodeUnits": [],
        "GravityScheme": [],
        "Parameters": [],
        "Header": ["PartTypeNames"],
        "SubgridScheme": ["GrainToElementMapping", "NamedColumns"],
    }

    # Convert header name to byte string
    header_name = str.encode(config.header_name)

    # Perform concatenation
    concatenate_snapshots(
        paths=paths,
        filename=config.output_name,
        groups=relevant_groups,
        name=header_name,
    )
