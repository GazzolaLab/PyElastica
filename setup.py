#!/usr/bin/env python
# Thanks and credits to https://github.com/navdeep-G/setup.py for setup.py format
import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "pyelastica"
DESCRIPTION = "Elastica is a software to simulate the dynamics of filaments that, at every cross-section, can undergo all six possible modes of deformation, allowing the filament to bend, twist, stretch and shear, while interacting with complex environments via muscular activity, surface contact, friction and hydrodynamics."
URL = "https://github.com/GazzolaLab/PyElastica"
EMAIL = "armant2@illinois.edu"
AUTHOR = "GazzolaLab"
REQUIRES_PYTHON = ">=3.5.0"
VERSION = "0.2.0.post1"

# What packages are required for this module to be executed?
REQUIRED = ["numpy>=1.19.2", "matplotlib>=3.3.2", "scipy>=1.5.2", "tqdm>=4.61.1", "numba==0.51.0"]

# What packages are optional?
EXTRAS = {
    "Optimization and Inverse Design": ["cma"],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    #packages=["elastica"],
    package_dir={"elastica": "./elastica"},
    packages=find_packages(),
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    download_url="https://github.com/GazzolaLab/PyElastica/archive/refs/tags/0.2.0.post1.tar.gz",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    # $ setup.py publish support: We don't know how to use this part. We already know how to publish.!
    #cmdclass={
    #    'upload': UploadCommand,
    #},
)
