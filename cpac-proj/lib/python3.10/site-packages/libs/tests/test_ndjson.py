# coding: utf-8

from __future__ import unicode_literals, absolute_import, print_function, division

import os
from unittest import TestCase

from libs.iolib.file import create_temp_dir, remove_temp_dir
from libs.ndjson import items_to_ndjsongz, ndjsongz_to_items


class NdjsongzTest(TestCase):

    def test_ndjsongz(self):
        temp_dir = create_temp_dir()
        try:
            file_path = os.path.join(temp_dir, 'test.ndjson.gz')
            items = [{"id": 1}, {"id": 2}]
            items_to_ndjsongz(items, file_path)
            new_items = list(ndjsongz_to_items(file_path))
            self.assertListEqual(items, new_items)
        finally:
            remove_temp_dir(temp_dir)
