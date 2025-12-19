import importlib.metadata

try:
    VERSION = importlib.metadata.version("pyelastica")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"
