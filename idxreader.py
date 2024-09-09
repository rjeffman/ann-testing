# SPDX: gpl-v3-or-later
"""Implement an IDX3 file reader and generator."""

import struct
from functools import reduce

DATA_TYPE = {
    0x08: ("B", 1),
    0x09: ("b", 1),
    0x0B: ("h", 2),
    0x0C: ("i", 4),
    0x0D: ("f", 4),
    0x0E: ("d", 8),
}


class IDXFile:
    """Reads an IDX file"""
    def __init__(self, file):
        """Generator for data in an IDX file."""
        # IDX file stores data in MSB format.
        self.file = file
        # Magic number 0x0000
        (zero,) = struct.unpack(">h", file.read(2))
        assert zero == 0x0000
        # Data type (1 byte)
        self.data_type, self.word_size = DATA_TYPE.get(
            struct.unpack(">b", file.read(1))[0], (None, 0)
        )
        assert self.data_type is not None
        # Number of dimensions (num items + each dimension) (1 byte)
        (dimensions,) = struct.unpack(">b", file.read(1))
        # Size of each dimension (4 byte MSB integer)
        self.dim_sizes = struct.unpack(
            f">{'i'*dimensions}",
            file.read(4*dimensions)
        )
        # adjust data size
        self.items, self.strip = (
            (self.dim_sizes[0], self.word_size) if dimensions == 1
            else (
                self.dim_sizes[0],
                self.word_size * reduce(
                    lambda x, y: x * y, self.dim_sizes[1:]
                )
            )
        )

    def __next__(self):
        """Yield each object."""
        if self.items > 0:
            data = struct.unpack(
                f">{self.data_type*self.strip}",
                self.file.read(self.strip)
            )
            self.items -= 1
            return data
        raise StopIteration()

    def __iter__(self):
        """Make object iterable."""
        return self

    @property
    def dimensions(self):
        """Return the number of dimensions for the data."""
        return self.dim_sizes[1:]


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "br") as idx3_file:
        data_file = IDXFile(idx3_file)
        mask = (1 << data_file.word_size*8) - 1
        for i, values in enumerate(data_file):
            print(i + 1, "values:", [x/mask for x in values])
