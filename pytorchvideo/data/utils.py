# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import csv
import itertools
import logging
import math
import threading
from collections import defaultdict
from dataclasses import Field, field as dataclass_field, fields as dataclass_fields
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr


logger = logging.getLogger(__name__)


def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)


def secs_to_pts(time_in_seconds: float, time_base: float, start_pts: float) -> float:
    """
    Converts a time (in seconds) to the given time base and start_pts offset
    presentation time.

    Returns:
        pts (float): The time in the given time base.
    """
    if time_in_seconds == math.inf:
        return math.inf

    time_base = float(time_base)
    return int(time_in_seconds / time_base) + start_pts


def pts_to_secs(time_in_seconds: float, time_base: float, start_pts: float) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.
    """
    if time_in_seconds == math.inf:
        return math.inf

    return (time_in_seconds - start_pts) * float(time_base)


class MultiProcessSampler(torch.utils.data.Sampler):
    """
    MultiProcessSampler splits sample indices from a PyTorch Sampler evenly across
    workers spawned by a PyTorch DataLoader.
    """

    def __init__(self, sampler: torch.utils.data.Sampler) -> None:
        self._sampler = sampler

    def __iter__(self):
        """
        Returns:
            Iterator for underlying PyTorch Sampler indices split by worker id.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 0:

            # Split sampler indexes by worker.
            video_indexes = range(len(self._sampler))
            worker_splits = np.array_split(video_indexes, worker_info.num_workers)
            worker_id = worker_info.id
            worker_split = worker_splits[worker_id]
            if len(worker_split) == 0:
                logger.warning(
                    f"More data workers({worker_info.num_workers}) than videos"
                    f"({len(self._sampler)}). For optimal use of processes "
                    "reduce num_workers."
                )
                return iter(())

            iter_start = worker_split[0]
            iter_end = worker_split[-1] + 1
            worker_sampler = itertools.islice(iter(self._sampler), iter_start, iter_end)
        else:

            # If no worker processes found, we return the full sampler.
            worker_sampler = iter(self._sampler)

        return worker_sampler


def optional_threaded_foreach(
    target: Callable, args_iterable: Iterable[Tuple], multithreaded: bool
):
    """
    Applies 'target' function to each Tuple args in 'args_iterable'.
    If 'multithreaded' a thread is spawned for each function application.

    Args:
        target (Callable):
            A function that takes as input the parameters in each args_iterable Tuple.

        args_iterable (Iterable[Tuple]):
            An iterable of the tuples each containing a set of parameters to pass to
            target.

        multithreaded (bool):
            Whether or not the target applications are parallelized by thread.
    """

    if multithreaded:
        threads = []
        for args in args_iterable:
            thread = threading.Thread(target=target, args=args)
            thread.start()
            threads.append(thread)

        for t in threads:  # Wait for all threads to complete
            t.join()
    else:
        for args in args_iterable:
            target(*args)


class DataclassFieldCaster:
    """
    Class to allow subclasses wrapped in @dataclass to automatically
    cast fields to their relevant type by default.

    Also allows for an arbitrary intialization function to be applied
    for a given field.
    """

    COMPLEX_INITIALIZER = "DataclassFieldCaster__complex_initializer"

    def __post_init__(self) -> None:
        f"""
        This function is run by the dataclass library after '__init__'.

        Here we use this to ensure all fields are casted to their declared types
        and to apply any complex field_initializer functions that have been
        declared via the 'complex_initialized_dataclass_field' method of
        this class.

        A complex field_initializer for a given field would be stored in the
        field.metadata dictionary at:
            key = '{self.COMPLEX_INITIALIZER}' (self.COMPLEX_INITIALIZER)

        """
        for field in dataclass_fields(self):
            value = getattr(self, field.name)
            # First check if the datafield has been set to the declared type or
            # if the datafield has a declared complex field_initializer.
            if (
                not isinstance(value, field.type)
                or DataclassFieldCaster.COMPLEX_INITIALIZER in field.metadata
            ):
                # Apply the complex field_initializer function for this field's value,
                # assert that the resultant type is the declared type of the field.
                if DataclassFieldCaster.COMPLEX_INITIALIZER in field.metadata:
                    setattr(
                        self,
                        field.name,
                        field.metadata[DataclassFieldCaster.COMPLEX_INITIALIZER](value),
                    )
                    assert isinstance(getattr(self, field.name), field.type), (
                        f"'field_initializer' function of {field.name} must return "
                        f"type {field.type} but returned type {type(getattr(self, field.name))}"
                    )
                else:
                    # Otherwise attempt to cast the field's value to its declared type.
                    setattr(self, field.name, field.type(value))

    @staticmethod
    def complex_initialized_dataclass_field(
        field_initializer: Callable, **kwargs
    ) -> Field:
        """
        Allows for the setting of a function to be called on the
        named parameter associated with a field during initialization,
        after __init__() completes.

        Args:
            field_initializer (Callable):
                The function to be called on the field

            **kwargs: To be passed downstream to the dataclasses.field method

        Returns:
            (dataclasses.Field) that contains the field_initializer and kwargs infoÎ
        """
        metadata = kwargs.get("metadata") or {}
        assert DataclassFieldCaster.COMPLEX_INITIALIZER not in metadata
        metadata[DataclassFieldCaster.COMPLEX_INITIALIZER] = field_initializer
        kwargs["metadata"] = metadata
        return dataclass_field(**kwargs)


def load_dataclass_dict_from_csv(
    input_csv_file_path: str,
    dataclass_class: type,
    dict_key_field: str,
    list_per_key: bool = False,
) -> Dict[Any, Union[Any, List[Any]]]:
    """
    Args:
        input_csv_file_path (str): File path of the csv to read from
        dataclass_class (type): The dataclass to read each row into.
        dict_key_field (str): The field of 'dataclass_class' to use as
            the dictionary key.
        list_per_key (bool) = False: If the output data structure
        contains a list of dataclass objects per key, rather than a
        single unique dataclass object.

    Returns:
        Dict[Any, Union[Any, List[Any]] mapping from the dataclass
        value at attr = dict_key_field to either:

        if 'list_per_key', a list of all dataclass objects that
        have equal values at attr = dict_key_field, equal to the key

        if not 'list_per_key', the unique dataclass object
        for which the value at attr = dict_key_field is equal to the key

    Raises:
        AssertionError: if not 'list_per_key' and there are
        dataclass obejcts with equal values at attr = dict_key_field
    """

    output_dict = defaultdict(list) if list_per_key else {}
    with g_pathmgr.open(input_csv_file_path) as dataclass_file:
        reader = csv.reader(dataclass_file, delimiter=",", quotechar='"')
        column_index = {header: i for i, header in enumerate(next(reader))}
        for line in reader:
            datum = dataclass_class(
                *(
                    line[column_index[field.name]]
                    for field in dataclass_fields(dataclass_class)
                )
            )
            dict_key = getattr(datum, dict_key_field)
            if list_per_key:
                output_dict[dict_key].append(datum)
            else:
                assert (
                    dict_key not in output_dict
                ), f"Multiple entries for {output_dict} in {dataclass_file}"
                output_dict[dict_key] = datum
    return output_dict


def save_dataclass_objs_to_headered_csv(
    dataclass_objs: List[Any], file_name: str
) -> None:
    """
    Saves a list of @dataclass objects to the specified csv file.

    Args:
        dataclass_objs (List[Any]):
            A list of @dataclass objects to be saved.

        file_name (str):
            file_name to save csv data to.
    """
    dataclass_type = type(dataclass_objs[0])
    field_names = [f.name for f in dataclass_fields(dataclass_type)]
    with g_pathmgr.open(file_name, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(field_names)
        for obj in dataclass_objs:
            writer.writerow([getattr(obj, f) for f in field_names])
