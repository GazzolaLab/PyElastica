import importlib.metadata

try:
    VERSION = importlib.metadata.version("elastica")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"
