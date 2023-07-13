#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import os
import glob
import shutil
import string
import os.path as op
import pathlib


class Filepath(pathlib.Path):
    _flavour = type(pathlib.Path())._flavour
    SAFE = "".join((string.ascii_letters, string.digits, " ", "_", "-", "*"))

    def __new__(cls, *args, ext=None):
        path = "/".join([str(arg) for arg in args])
        path = "/".join(op.normpath(path).split("\\")).split("/")
        path = [cls.stabilize(s) for s in path]
        path[-1] = (".".join((path[-1], ext)) if ext is not None else path[-1])
        path = "/".join(path)
        return super(Filepath, cls).__new__(cls, path)

    def __init__(self, *args, ext=None):
        super().__init__()
        self.path = str(self).split("/")
        self.ext = ext

    def __repr__(self):
        return "/".join(self.path)

    def __contains__(self, item):
        return item in str(self)

    def __len__(self):
        return len(self.path)

    def __bool__(self):
        func = (op.isfile if "." in self else op.isdir)
        return func(repr(self))

    def __getitem__(self, idx):
        path = self.path[idx]
        path = ([path] if type(path) != list else path)
        return Filepath("/".join(path))

    def __setitem__(self, idx, value):
        value, ext = Filepath.stabilize(value), Filepath.extension(value)
        value = (
            ".".join((value, ext))
            if ext is not None and idx + 1 == len(self) else value)
        self.path[idx] = value

    def __delitem__(self, idx):
        del self.path[idx]

    def __add__(self, other):
        return Filepath(str(self) + str(other))

    def __radd__(self, other):
        return Filepath(str(other) + str(self))

    def astype(self, ext):
        return Filepath(repr(self), ext=ext)

    def search(self, recursive=False, first=False):
        path = sorted(glob.glob(repr(self), recursive=recursive))
        path = (None if len(path) == 0 else (
            Filepath(path[0], ext=self.ext) if first else [
                Filepath(s, ext=self.ext) for s in path]))
        return path

    def make(self):
        if not op.isdir(self.astype(None)):
            os.makedirs(str(self.astype(None)), exist_ok=False)

    def makeParent(self):
        self[:-1].make()

    def move(self, dst):
        dst = (
            Filepath(dst, self.extension(dst)) if type(dst) != Filepath
            else dst)
        if self and not dst:
            dst[:-1].make()
            shutil.move(src=self, dst=dst)
            self.path, self.ext = dst.path, dst.ext

    def remove(self):
        if self and op.isfile(self):
            os.remove(self)

    @staticmethod
    def extension(path):
        ext = (None if "." not in path else path[path.rindex(".") + 1:])
        return ext

    @staticmethod
    def stabilize(subpath):
        """
        Remove unstable characters from proposed filename

        @param subpath: filename to clean
        @type subpath: string

        @return: filename with unstable characters replaced with "-"
        @rtype: string
        """
        subpath = str(subpath)
        subpath = (
            subpath[:subpath.rindex(".")] if "." in subpath else subpath)
        subpath = list(subpath)
        for i in range(len(subpath)):
            subpath[i] = (subpath[i] if subpath[i] in Filepath.SAFE else "-")

        return "".join(subpath)
