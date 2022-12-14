# coding: utf-8

from __future__ import unicode_literals, absolute_import, print_function, division

import os
from unittest import TestCase

from libs.iolib.file import create_temp_dir, remove_temp_dir, write_file, read_file


class IolibFileTest(TestCase):

    def test_temp(self):
        temp_dir = create_temp_dir()
        try:
            file_path = os.path.join(temp_dir, 'test.txt')
            write_file(file_path, temp_dir)
            content = read_file(file_path)
            self.assertEqual(content, temp_dir)
        finally:
            remove_temp_dir(temp_dir)
