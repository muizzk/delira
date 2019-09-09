import unittest
import numpy as np

from delira.utils.codecs import Encoder, Decoder
#TODO: skips


class CodecsTest(unittest.TestCase):
    def test_encoder(self):
        test_dict = {}
        test_dict['number'] = 1
        test_dict['string'] = "test_string"
        test_dict['list'] = [0, 1, 2, "skjd"]
        test_dict['dict'] = {"key0": 0, "key1": 1, "key2": 2}
        test_dict['tuple'] = (1, 2, 3)
        test_dict['none'] = None
        test_dict['nparray'] = np.array([0, 1, 2])
        test_dict['function'] = np.min
        test_dict['class'] = np.ndarray

        encoded_test_dict = Encoder().encode(test_dict)

        self.assertTrue(encoded_test_dict['number'] == 1)
        self.assertTrue(encoded_test_dict['string'] == "test_string")
        self.assertListEqual(encoded_test_dict['list'], [0, 1, 2, "skjd"])
        self.assertDictEqual(encoded_test_dict['dict'], {
                             "key0": 0, "key1": 1, "key2": 2})
        self.assertDictEqual(encoded_test_dict['tuple'], {
            "__convert__": {
                "repr": [1, 2, 3],
                "type": {
                    "__type__": {"module": "builtins", "name": "tuple"}}
            }})
        self.assertIsNone(encoded_test_dict["none"])
        self.assertDictEqual(encoded_test_dict["nparray"],
                             {"__array__": [0, 1, 2]})
        self.assertDictEqual(encoded_test_dict["function"], {
            "__function__": {"module": "numpy", "name": "amin"}})
        self.assertDictEqual(encoded_test_dict["class"], {
            "__type__": {"module": "numpy", "name": "ndarray"}})

    def test_decoder(self):
        test_dict = {}
        test_dict['number'] = 1
        test_dict['string'] = "test_string"
        test_dict['list'] = [0, 1, 2, "skjd"]
        test_dict['dict'] = {"key0": 0, "key1": 1, "key2": 2}
        test_dict['tuple'] = {"__convert__": {
            "repr": [1, 2, 3],
            "type": {"__type__": {"module": "builtins", "name": "tuple"}}
        }}
        test_dict['none'] = None
        test_dict['nparray'] = {"__array__": [0, 1, 2]}
        test_dict['function'] = {"__function__": {
            "module": "numpy", "name": "amin"}}
        test_dict['class'] = {"__type__": {
            "module": "numpy", "name": "ndarray"}}
        test_dict["classargs"] = {"__classargs__":
                                      {"module": "numpy",
                                       "name": "ndarray",
                                       "args": [[1, 2, 3]]
                                       }
                                  }
        test_dict["funcargs"] = {"__functionargs__" :
                                     {"module": "numpy",
                                      "name": "min",
                                      "kwargs": {"axis": (1, 2)}}
                                 }

        decoded_dict = Decoder().decode(test_dict)

        self.assertTrue(decoded_dict['number'] == 1)
        self.assertTrue(decoded_dict['string'] == "test_string")
        self.assertListEqual(decoded_dict['list'], [0, 1, 2, "skjd"])
        self.assertDictEqual(decoded_dict['dict'], {
                             "key0": 0, "key1": 1, "key2": 2})
        self.assertTupleEqual(decoded_dict['tuple'], (1, 2, 3))
        self.assertIsNone(decoded_dict["none"])
        self.assertTrue((decoded_dict["nparray"] == np.array([0, 1, 2])).all())
        self.assertTrue(
            decoded_dict["function"].__module__ == np.min.__module__)
        self.assertTrue(
            decoded_dict["function"].__name__ == np.min.__name__)
        self.assertTrue(
            decoded_dict["class"].__module__ == np.ndarray.__module__)
        self.assertTrue(
            decoded_dict["class"].__name__ == np.ndarray.__name__)
        self.assertTrue(test_dict["classargs"].shape == (1, 2, 3))
        self.assertTrue(test_dict["funcargs"].args[0] == [])
        self.assertTrue(test_dict["funcargs"].args[1]["axis"] == (1, 2))


if __name__ == '__main__':
    unittest.main()
