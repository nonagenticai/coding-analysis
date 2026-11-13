import importlib

DEFAULT = importlib.import_module("app.tasks")


def load(name):
    return importlib.import_module(name)
