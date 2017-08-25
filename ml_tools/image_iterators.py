import threading
from itertools import repeat
from copy import copy


class _IteratorWithFixedIndexes(object):
    """This is a wrapper around a keras.preprocessing.image.Iterator
       with an index_generator that always returns the same indexes
    """

    def __init__(self, iterator, indexes):
        self.iterator = copy(iterator)
        self.iterator.index_generator = repeat(indexes)

    def next(self):
        return self.iterator.next()

    def __next__(self):
        return self.next()


class MultiDirectoryIterator(object):
    """This is basically a (thread-safe!) zip() for keras.preprocessing.image.Iterators"""

    def __init__(self, iterators):
        self.iterators = iterators
        self.samples = iterators[0].samples
        self.batch_size = iterators[0].batch_size
        self.target_sizes = [it.target_size for it in iterators]
        self.lock = threading.Lock()

    def next(self):
        # first, we fetch new indexes for each iterator
        # under lock to avoid a race condition
        with self.lock:
            new_indexes = [next(it.index_generator) for it in self.iterators]

        # now, we make iterator wrappers that use those indexes
        iterator_wrappers = [_IteratorWithFixedIndexes(it, idx) for it, idx in zip(self.iterators, new_indexes)]

        # now we can call the iterators safely
        batches_x = []
        for it in iterator_wrappers:
            batch_x, y = it.next()
            batches_x.append(batch_x)
        return batches_x, y

    def __next__(self):
        return self.next()
