# coding: utf-8

from __future__ import unicode_literals, absolute_import, print_function, division

import json
import gzip


def items_to_ndjsongz(items, file_path):
    """ Write items to ndjsongz
    """
    with gzip.open(file_path, 'wb') as f:
        for item in items:
            line = json.dumps(item, ensure_ascii=False) + '\n'
            f.write(line.encode('utf-8'))


def ndjsongz_to_items(file_path):
    """ Read ndjsongz to items
    """
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            if line:
                yield json.loads(line)
