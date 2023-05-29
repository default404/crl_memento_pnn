import os, sys, inspect
__PRJ_DIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
if not __PRJ_DIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __PRJ_DIR__)

del inspect