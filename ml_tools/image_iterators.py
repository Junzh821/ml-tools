import threading


class MultiDirectoryIterator(object):
    def __init__(self, iterators):
        self.iterators = iterators
        self.samples = iterators[0].samples
        self.lock = threading.Lock()

    def next(self):
        # this lock is pretty wasteful but there's not much choice with the way DirectoryIterator is set up
        # see https://github.com/fchollet/keras/blob/2d8739dda9859b91bf2b7da5402d242555c48d7d/keras/preprocessing/image.py#L1022
        with self.lock:
            batches_x = []
            for it in self.iterators:
                batch_x, y = it.next()
                batches_x.append(batch_x)
        return batches_x, y

    def __next__(self):
        return self.next()
