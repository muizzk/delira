from collections import Iterable
from delira.data_loading.transforms.abstract_transform import AbstractTransform


class Compose(AbstractTransform):
    def __init__(self, transform, *transforms):
        if not isinstance(transform, Iterable):
            transform = [transform]
        else:
            transform = list(transform)

        transform.extend(transforms)

        self._transforms = tuple(transforms)

    def __call__(self, **data_dict):
        for trafo in self._transforms:
            data_dict = trafo(**data_dict)

        return data_dict
