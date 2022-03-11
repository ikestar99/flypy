#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import os
import shutil
import string
import os.path as op
from glob import glob as gg


ALLOWED = "".join((string.ascii_letters, string.digits, " ", "_", "-"))


def getPath(*args, ext=None):
    """
    Convert arbitrary number of filenames or subpaths into one filepath

    @param args: filenames or filepaths to be combined, in order
    @type args: arguments
    @param ext: file extension, if filepath
    @type ext: string

    @return: path to desired file or directory
    @rtype: string
    """
    path = op.normpath(op.join(*[
        op.join(*op.join(*arg.split("\\")).split("/")) for arg in args]))    
    path = (".".join([path, ext]) if ext is not None else path)
    return "".join(("/", path))


def getParent(path, num=1):
    """
    Find parent directory of a file or subdirectory

    @param path: filepath or subdirectory from which to find a parent
    @type path: string
    @param num: how many subdirectories to traverse before returning parent
    @type num: int

    @return: path to parent directory
    @rtype: string
    """
    for x in range(num):
        if "/" in path:
            path = path[:path.rindex("/")]

    return path


def getName(path):
    """
    Find name of a file or subdirectory

    @param path: filepath or subdirectory from which to find a name
    @type path: string

    @return: name of given file or subdirectory
    @rtype: string
    """
    name = "".join(("/", path))
    name = name[name.rindex("/") + 1:]
    return name


def cleanName(filename):
    """
    Remove unstable characters from proposed filename

    @param filename: filename to clean
    @type filename: string

    @return: filename with unstable characters replaced with "-"
    @rtype: string
    """
    filename = list(str(filename))
    for idx in range(len(filename)):
        if filename[idx] not in ALLOWED:
            filename[idx] = "-"

    filename = "".join(filename)
    return filename


def changeExt(path, ext=None):
    """
    Add or change the extension of a filepath

    @param path: filepath to modify
    @type path: string
    @param ext: extension to add to filepath
    @type ext: string

    @return: filepath with old extension removed and any new extension added
    @rtype: string
    """
    path = (path[:path.rindex(".")] if "." in path else path)
    if ext is not None:
        path = ".".join([path, ext])

    return path


def makeDir(path):
    """
    If a necessary directory does not exist, create it

    @param path: path to necessary directory
    @type path: string
    """
    if not op.isdir(path):
        os.makedirs(path, exist_ok=False)


def makeParent(path):
    """
    If a necessary parent directory does not exist, create it

    @param path: filepath for which to check parent
    @type path: string
    """
    makeDir(getParent(path))


def removeParent(path):
    """
    If parent directory of filepath is empty, delete it

    @param path: filepath for which to check parent
    @type path: string
    """
    parent = getParent(path)
    if op.isdir(parent) and not os.listdir(parent):
        shutil.rmtree(parent)


def movePath(src, dst):
    """
    Move a filepath or subdirectory to a new location

    @param src: path to file or subdirectory to be moved
    @type src: string
    @param dst: desired path to file or subdirectory
    @type dst: string
    """
    if all(((type(src) == str), (type(dst) == str),
            (not any(((op.isfile(dst)), (op.isdir(dst))))),
            (op.isfile(src) or op.isdir(src)))):
        makeParent(dst)
        shutil.move(src=src, dst=dst)


def glob(*args, ext=None):
    """
    glob function from glob package with following additions:
        - return any matches in alphabetical order
        - return None if no matches found

    @param args: passed to getPath()
    @type args: arguments
    @param ext: passed to getPath()
    @type ext: string

    @return: list of paths that match desired pattern
    @rtype: list
    """
    path = getPath(*args, ext=ext)
    pathList = sorted(gg(path))
    pathList = (pathList if len(pathList) > 0 else None)
    return pathList


def recursiveGlob(*args, ext=None):
    """
    glob with ability to search in subdirectories

    @param args: passed to getPath()
    @type args: arguments
    @param ext: passed to getPath()
    @type ext: string

    @return: list of paths that match desired pattern
    @rtype: list
    """
    path = getPath(*args, ext=ext)
    pathList = sorted(gg(path, recursive=True))
    pathList = (pathList if len(pathList) > 0 else None)
    return pathList


def firstGlob(*args, ext=None):
    """
    glob but only return first match

    @param args: passed to getPath()
    @type args: arguments
    @param ext: passed to getPath()
    @type ext: string

    @return: firat path that matches desired pattern
    @rtype: string
    """
    path = getPath(*args, ext=ext)
    path = (gg(path, recursive=True) if "**" in args else gg(path))
    path = (path[0] if len(path) > 0 else None)
    return path
