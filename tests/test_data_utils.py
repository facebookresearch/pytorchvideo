import contextlib
import csv
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path

from pytorchvideo.data.utils import DataclassFieldCaster, load_dataclass_dict_from_csv


@dataclass
class TestDataclass(DataclassFieldCaster):
    a: str
    b: int
    b_plus_1: int = DataclassFieldCaster.complex_initialized_dataclass_field(
        lambda v: int(v) + 1
    )
    c: float
    d: list
    e: dict = DataclassFieldCaster.complex_initialized_dataclass_field(lambda v: {v: v})


@dataclass
class TestDataclass2(DataclassFieldCaster):
    a: str
    b: int


@contextlib.contextmanager
def write_data_class_to_csv(dataclass_objs):
    assert dataclass_objs
    with tempfile.TemporaryDirectory(prefix=f"{TestDataUtils}") as tempdir:
        dataclass_type = type(dataclass_objs[0])
        field_names = [f.name for f in dataclass_fields(dataclass_type)]
        file_name = Path(tempdir) / "data.csv"
        with open(file_name, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"')
            writer.writerow(field_names)
            for obj in dataclass_objs:
                writer.writerow([getattr(obj, f) for f in field_names])
        yield file_name


class TestDataUtils(unittest.TestCase):
    def test_DataclassFieldCaster(self):
        test_obj = TestDataclass("1", "1", "1", "1", "abc", "k")

        self.assertEqual(test_obj.a, "1")
        self.assertEqual(type(test_obj.a), str)

        self.assertEqual(test_obj.b, 1)
        self.assertEqual(type(test_obj.b), int)
        self.assertEqual(test_obj.b_plus_1, 2)

        self.assertEqual(test_obj.c, 1.0)
        self.assertEqual(type(test_obj.c), float)

        self.assertEqual(test_obj.d, ["a", "b", "c"])
        self.assertEqual(type(test_obj.d), list)

        self.assertEqual(test_obj.e, {"k": "k"})
        self.assertEqual(type(test_obj.e), dict)

    def test_load_dataclass_dict_from_csv_value_dict(self):
        objects = [
            TestDataclass2("a", 1),
            TestDataclass2("b", 2),
            TestDataclass2("c", 3),
            TestDataclass2("d", 4),
        ]
        with write_data_class_to_csv(objects) as csv_file_name:
            test_dict = load_dataclass_dict_from_csv(
                csv_file_name, TestDataclass2, "a", list_per_key=False
            )
            self.assertEqual(len(test_dict), 4)
            self.assertEqual(test_dict["c"].b, 3)

    def test_load_dataclass_dict_from_csv_list_dict(self):
        objects = [
            TestDataclass2("a", 1),
            TestDataclass2("a", 2),
            TestDataclass2("b", 3),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
        ]
        with write_data_class_to_csv(objects) as csv_file_name:
            test_dict = load_dataclass_dict_from_csv(
                csv_file_name, TestDataclass2, "a", list_per_key=True
            )
            self.assertEqual(len(test_dict), 3)
            self.assertEqual([x.b for x in test_dict["a"]], [1, 2])
            self.assertEqual([x.b for x in test_dict["b"]], [3])
            self.assertEqual([x.b for x in test_dict["c"]], [4, 4, 4])

    def test_load_dataclass_dict_from_csv_throws(self):
        objects = [
            TestDataclass2("a", 1),
            TestDataclass2("a", 2),
            TestDataclass2("b", 3),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
        ]
        with write_data_class_to_csv(objects) as csv_file_name:
            self.assertRaises(
                AssertionError,
                lambda: load_dataclass_dict_from_csv(
                    csv_file_name, TestDataclass2, "a", list_per_key=False
                ),
            )
