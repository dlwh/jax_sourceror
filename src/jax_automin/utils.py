from collections.abc import MutableSet


class IdentitySet(MutableSet):
    """Set that compares objects by identity.

    This is a set that compares objects by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable set, but it does not support the ``__hash__`` method and
    therefore cannot be used as a dictionary key or as an element of another
    set.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, value):
        return id(value) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[id(value)] = value

    def discard(self, value):
        self._data.pop(id(value), None)

    def __repr__(self):
        return f"IdentitySet({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentitySet({list(str(x) for x in self._data.values())})"