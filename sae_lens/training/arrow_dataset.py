import json
from pathlib import Path


import pyarrow as pa
import torch

import tqdm
from tqdm.contrib.concurrent import process_map


def worker(file_path: Path) -> int:
    with pa.memory_map(str(file_path), "rb") as f:
        num_rows = pa.ipc.open_stream(f).read_all().num_rows
    return num_rows


class ArrowDataset:
    def __init__(
        self,
        path: str | Path,
        columns: list[str],
        device: str = "cpu",
        dtype: str = "float32",
        assume_equal_splits: bool = False,
    ):
        """
        Initialize the ArrowDataset class.

        Args:
            path (str | Path): Path to the dataset file.
        """
        self.path = Path(path)
        self.device = device
        self.dtype = dtype
        self.assume_equal_splits = assume_equal_splits

        with open(self.path / "dataset_info.json", "rt") as f:
            self.dataset_info = json.load(f)

        with open(self.path / "state.json", "rt") as f:
            state_dict = json.load(f)
            relative_data_entries = state_dict.get("_data_files")

        if not isinstance(relative_data_entries, list) or len(relative_data_entries) == 0:
            raise ValueError(f"Invalid _data_files in {self.path / 'state.json'}: {relative_data_entries}")

        self.data_files = [
            self.path / entry.get('filename') for entry in relative_data_entries
        ]

        self.columns = self.dataset_info.get("features", dict())
        self.column_names = list(self.columns.keys())

        self._current_loaded_file_idx = None
        self._current_loaded_data = None
        self._offsets = None

    def _calculate_all_offsets(self, assume_equal_splits: bool = False):
        """Calculate offsets for all files in the dataset."""
        if self._offsets is not None:
            return

        curr_idx = 0
        self._offsets = []

        if assume_equal_splits:
            iterator = [0, -1]
            if tqdm is not None:
                iterator = tqdm.tqdm(iterator, desc="Calculating offsets")

            lengths = [
                self._get_table(i).num_rows for i in iterator
            ]

            length_of_first, length_of_last = lengths[0], lengths[1]

            for i in range(len(self.data_files)):
                self._offsets.append(curr_idx)
                curr_idx += length_of_first

            total_length = curr_idx - length_of_first + length_of_last

        else:
            row_counts = process_map(
                worker,
                self.data_files,
                max_workers=8,
                desc="Calculating offsets",
                total=len(self.data_files),
            )

            for count in row_counts:
                self._offsets.append(curr_idx)
                curr_idx += count

            total_length = curr_idx

        self._offsets.append(total_length)

    def __len__(self):
        if self._offsets is None:
            self._calculate_all_offsets(self.assume_equal_splits)
        return self._offsets[-1]

    def __getitem__(self, idx: tuple[str, int | slice]) -> torch.Tensor:

        column_name, _idx = idx
        if column_name not in self.column_names:
            raise ValueError(f"Column {column_name} not found in dataset")

        if isinstance(_idx, slice):
            start = _idx.start or 0
            stop = _idx.stop or len(self)
            step = _idx.step or 1

            if start < 0:
                start += len(self)
            if stop < 0:
                stop += len(self)

            if start < 0 or stop > len(self) or step <= 0:
                raise IndexError("Slice out of range")

            indices = list(range(start, stop, step))
        elif isinstance(_idx, int):
            if _idx < 0:
                _idx += len(self)
            if _idx < 0 or _idx >= len(self):
                raise IndexError("Index out of range")
            indices = [_idx]
        else:
            raise TypeError("Slice must be an int or a slice")

        base_shape = tuple(self.columns[column_name].get('shape', 1))
        shape = (len(indices), *base_shape)

        if sum(d != 1 for d in base_shape) != 1:
            raise ValueError("Currently only support a single non-1 dimension in the shape of the data.")

        data = torch.empty(shape, dtype=self.dtype, device=self.device)

        file_idxs_to_load = []
        for i, start in enumerate(self._offsets[:-1]):
            if indices[0] < self._offsets[i + 1] and indices[-1] >= start:
                file_idxs_to_load.append(i)
            if indices[-1] < self._offsets[i + 1]:
                break

        # If our slice includes any indices from the currently loaded file,
        # we should read those indices first before loading the next file.
        if self._current_loaded_file_idx in file_idxs_to_load:
            file_idxs_to_load.remove(self._current_loaded_file_idx)
            file_idxs_to_load.insert(0, self._current_loaded_file_idx)

        for file_idx in file_idxs_to_load:
            file_start = self._offsets[file_idx]
            file_end = self._offsets[file_idx + 1]

            current_indices, row_numbers, data_indices = zip(*(
                (i, i - file_start, data_idx)
                for data_idx, i in enumerate(indices)
                if file_start <= i < file_end
            ))
            rows = self._get_rows(file_idx, row_numbers)
            for row, data_idx in zip(rows.columns[0], data_indices):
                data[data_idx, :] = torch.from_numpy(row[0].values.to_numpy()).to(dtype=self.dtype, device=self.device)

        data.squeeze_()

        return data

    def _get_rows(self, file_idx: int, rows: list[int]) -> pa.Table:
        """
        Get specific rows from a file.

        Args:
            file_idx (int): Index of the file to read from.
            indices (list[int]): List of indices to retrieve.

        Returns:
            pa.Table: Table containing the requested rows.
        """
        table = self._get_table(file_idx)

        return table.take(rows)

    def _get_table(self, file_idx: int) -> pa.Table:
        """
        Get the entire table from a file.

        Args:
            file_idx (int): Index of the file to read from.

        Returns:
            pa.Table: Table containing all rows.
        """
        if file_idx < 0:
            file_idx += len(self.data_files)

        if file_idx < 0 or file_idx >= len(self.data_files):
            raise IndexError("File index out of range")

        if file_idx != self._current_loaded_file_idx:
            self._current_loaded_file_idx = file_idx
            with pa.memory_map(str(self.data_files[file_idx]), "rb") as f:
                self._current_loaded_data = pa.ipc.open_stream(f).read_all()

        return self._current_loaded_data
