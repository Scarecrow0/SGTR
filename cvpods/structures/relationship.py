# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Union

import torch

from cvpods.structures.instances import Instances


class Relationships:
    """
    This structure stores the Instances and their paired relationships as the 
    for represent the relationship and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    """

    def __init__(self, instances: Instances, rel_pair_tensor: torch.Tensor, **kwargs: Any):
        """
        Args:
            bbox (Boxes): the default Boxes classes instance
            rel_pair_tensor (Tensor[long]): 
                a Nx2 matrix.  Each row is index of subject and object instance in Instances.
        """
        device = rel_pair_tensor.device
        rel_pair_tensor = torch.as_tensor(rel_pair_tensor, device=device).long()

        self._fields: Dict[str, Any] = {
            "instances": instances,
            "rel_pair_tensor": rel_pair_tensor,
        }

        self._meta_info: Dict[str, Any] = {
        }
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Relationships! "
                                 .format(name, ))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                    len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    def fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return list(self._fields.keys())

    def to(self, device: str) -> "Relationships":
        for k, v in self._fields.items():
            if hasattr(self._fields[k], "to"):
                self._fields[k] = self._fields[k].to(device)

        return self

    def compress_tensor(self):
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float:
                    self._fields[k] = v.half()
                if v.dtype == torch.long:
                    self._fields[k] = v.int()
            if hasattr(v, 'compress_tensor'):
                self._fields[k] = self._fields[k].compress_tensor()

        return self
    
    def add_meta_info(self, k, v):
        self._meta_info[k] = v

    def meta_info_fields(self):
        return list(self._meta_info.keys())

    def get_meta_info(self, k):
        return self._meta_info[k]

    def has_meta_info(self, k):
        if self._meta_info.get(k) is not None:
            return True
        return False

    def update_instance(self, instances):
        assert isinstance(instances, Instances)
        assert (torch.max(self.rel_pair_tensor) - 1) <= len(instances)
        self._fields["instances"] = instances

    def get_rel_matrix(self, ) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        rel_matrix = torch.zeros((len(self.instances), len(self.instances)),
                                 device=self.rel_pair_tensor.device).long()
        rel_matrix[self.rel_pair_tensor[:, 0], self.rel_pair_tensor[:, 1]] = self.rel_label

        return rel_matrix

    def __len__(self) -> int:
        """use the number of relationship as the size of Relationship instance

        Returns:
            int: len of relationship triplets
        """
        return len(self.rel_pair_tensor)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Relationships":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Relationships(self.instances, self.rel_pair_tensor[item])
        for k, v in self._fields.items():
            if k == 'instances':
                continue
            ret.set(k, v[item])

        for k, v in self._meta_info.items():
            ret.add_meta_info(k, v)

        return ret

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"num_boxes={len(self.instances)},"
        s += f"num_rel={len(self.rel_pair_tensor)} "

        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, type(v))
        s += ')'

        return s

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.rel_pair_tensor.device


def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor
