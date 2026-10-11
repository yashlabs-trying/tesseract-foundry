from importlib.metadata import version as _v
try:
    __version__ = _v("tesseract-foundry")
except:
    __version__ = "1.0.0"
