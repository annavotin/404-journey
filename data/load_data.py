"""Convert datasets to unified format and normalised labels
Supported formats:
    - CoNLL format
    - WorldWideNews format

Output contains two files: data.parquet and labels.parquet
    - data.parquet contains the text data
    - labels.parquet contains the labels for each token in the text
"""
import argparse
from ner_evaluation.utils.ner_dataset import CoNLLDataset, WWNDataset


def create_dataset(inputstyle):
    """
    Create a dataset object based on the input style.
    Args:
        inputstyle (str): The style of the input dataset.
    Returns:
        NERDataset: An instance of the appropriate dataset class.
    """
    if inputstyle == 'conll':
        return CoNLLDataset()
    elif inputstyle == 'worldwidenews':
        return WWNDataset()
    else:
        raise ValueError(f"Unsupported input style: {inputstyle}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert dataset to CoNLL format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input dataset file.')
    parser.add_argument('--inputstyle', type=str,
                        choices=['conll', 'worldwidenews'],
                        required=True, help='Style of the input dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output data file.')
    args = parser.parse_args()

    # Create the dataset object based on the input style
    dataset = create_dataset(args.inputstyle)
    dataset.load_data(args.input)
    dataset.export_data(args.output)
