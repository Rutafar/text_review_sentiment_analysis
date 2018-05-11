
from tqdm import tqdm
from src.utils.utils import get_file_names
from src.data.make_first_raw_dataset import import_dataset, export_sampled_datasets


def file_pickling():
    files = get_file_names()
    for file in tqdm(files):
        test = import_dataset(file)
        export_sampled_datasets(test, file)


if __name__ == '__main__':
    file_pickling()