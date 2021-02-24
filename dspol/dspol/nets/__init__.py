import glob
import importlib
from os.path import basename, dirname, isfile, join

py_files = glob.glob(join(dirname(__file__), "*.py"))
importable = []
for f in py_files:
    if isfile(f) and not f.startswith("__") and not f.startswith("."):
        importable.append(basename(f)[:-3])
__all__ = importable

# Allow submodule usage with '.' operator
for im in importable:
    importlib.import_module(".{}".format(im), __package__)
