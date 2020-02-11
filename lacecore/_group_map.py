import numpy as np
import vg


class GroupMap:
    """
    An immutable map of possibly overlapping groups of elements.
    """

    def __init__(self, num_elements, group_names, masks, copy_masks=False):
        """
        Args:
            num_elements (int): The total number of elements. This determines
                the length of the masks.
            group_names (list): The names of the groups.
            masks (np.array): A boolean array with a row for each group and a
                column for each element.
        """
        if not isinstance(num_elements, int) or num_elements < 0:
            raise ValueError("num_elements should be a non-negative integer")
        if not all(isinstance(group_name, str) for group_name in group_names):
            raise ValueError("group_names should be a list of strings")
        vg.shape.check(locals(), "masks", (len(group_names), num_elements))
        if masks.dtype != np.bool:
            raise ValueError("Expected masks to be a bool array")

        if copy_masks:
            masks = masks.copy()
        masks.setflags(write=False)
        self._masks = masks
        self._group_names = {k: i for i, k in enumerate(group_names)}

    @classmethod
    def from_dict(cls, group_data, num_elements):
        """
        Args:
            group_data (dict): The group data. The keys are group names and
                the values are lists of element indices.
            num_elements (int): The total number of elements.
        """
        masks = np.zeros((len(group_data), num_elements), dtype=np.bool)
        for i, element_indices in enumerate(group_data.values()):
            try:
                masks[i][element_indices] = True
            except IndexError:
                raise ValueError(
                    "Element indices should be less than {}".format(num_elements)
                )
        return cls(
            num_elements=num_elements,
            group_names=group_data.keys(),
            masks=masks,
            copy_masks=False,
        )

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, group_name):
        """
        Return a read-only mask for the requested group.
        """
        try:
            index = self._group_names[group_name]
        except KeyError:
            raise KeyError("Unknown group: {}".format(group_name))
        return self._masks[index]

    def union(self, *group_names):
        """
        Return a writable mask containing the union of the requested groups.
        """
        indices = []
        invalid_group_names = []
        for group_name in group_names:
            try:
                indices.append(self._group_names[group_name])
            except KeyError:
                invalid_group_names.append(group_name)
        if len(invalid_group_names):
            raise KeyError("Unknown groups: {}".format(", ".join(invalid_group_names)))
        return np.any(self._masks[indices], axis=0)
