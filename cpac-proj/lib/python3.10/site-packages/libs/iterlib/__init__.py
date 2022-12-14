# coding: utf-8

import itertools


def uniqify(sequence, key=None) -> iter:
    """ uniqify

    Return uniqify sequence
    """
    seen = set()
    add = seen.add
    for x in sequence:
        value = key(x) if key else x
        if value in seen:
            continue
        add(value)
        yield x


def chunks(iterable, size=10):
    """ chunks

    Split list into multiple fixed size chunks
    """
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def isplit(lst, count=10):
    """ isplit

    Split list into n list
    """
    return (list(lst[i::count]) for i in range(count))
