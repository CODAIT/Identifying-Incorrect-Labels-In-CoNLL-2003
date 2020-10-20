import os
import glob
import json


def delete_lines(input_file, output_file, lines):
    """
    inputs:
        - input_file -> (Str) path to the input file
        - output_file -> (Str) path to the output file
        - lines -> list of line numbers to be deleted
    returns:
        - None
        writes a file in the same directory with a `_corrected` tag.
    Raises:
        - assertion error if a non-empty line was being deleted
    """
    with open(input_file, "r") as source_file:
        file_lines = source_file.readlines()
    for l in lines:
        assert file_lines[l] == "\n"
        del file_lines[l]
    with open(output_file, "w+") as new_file:
        for l in file_lines:
            new_file.write(l)


def process_dataset_file(dataset_fold, dataset_file, json_file, target_file=None):
    with open(json_file) as f:
        lines_to_delete = json.load(f)
    delete_lines(dataset_file, target_file, lines_to_delete[dataset_fold])
