# coding: utf-8

from __future__ import unicode_literals, absolute_import, print_function, division

from unittest import TestCase

from libs.iterlib import uniqify, chunks, isplit


class IterlibTest(TestCase):

    def test_uniqify(self):
        self.assertListEqual(list(uniqify([1, 2, 3])), [1, 2, 3])
        self.assertListEqual(list(uniqify([1, 2, 3, 3])), [1, 2, 3])
        self.assertListEqual(
            list(uniqify([{"id": 1}, {"id": 2}, {"id": 2}], key=lambda x: x["id"])),
            [{"id": 1}, {"id": 2}]
        )

    def test_chunks(self):
        self.assertListEqual(
            list(chunks(range(10), size=1)),
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        )
        self.assertListEqual(
            list(chunks(range(10), size=3)),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        )
        self.assertListEqual(
            list(chunks(range(10), size=10)),
            [list(range(10))]
        )
        self.assertListEqual(
            list(chunks(range(10), size=100)),
            [list(range(10))]
        )

    def test_isplit(self):
        self.assertListEqual(
            list(isplit(range(10), count=1)),
            [list(range(10))]
        )
        self.assertListEqual(
            list(isplit(range(10), count=4)),
            [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
        )
        self.assertListEqual(
            list(isplit(range(10), count=10)),
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        )
        self.assertListEqual(
            list(isplit(range(10), count=20)),
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [], [], [], [], [], [], [], [], [], []]
        )
