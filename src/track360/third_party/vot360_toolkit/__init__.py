"""Metric code of the official 360VOT toolkit (https://github.com/HuajianUP/360VOT).

``ope_benchmark.py`` and ``sphiou.py`` are copied from the toolkit's ``eval/``
directory with two changes that do not alter any value: ``sphiou`` is imported
relatively, and one assignment in ``SphIoU.interArea`` takes ``.item()`` so that it
runs on NumPy 2.  Known quirks are kept, so scores are the ones the benchmark reports.
"""
