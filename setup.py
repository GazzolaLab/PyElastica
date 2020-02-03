from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = "0.0.0"

setup(
    name="pyelastica",
    packages=['elastica, tests, examples'],
    package_dir={'elastica': './elastica'},
    version=version,
    description="Elastica is a software to simulate the dynamics of filaments that, at every cross-section, can undergo all six possible modes of deformation, allowing the filament to bend, twist, stretch and shear, while interacting with complex environments via muscular activity, surface contact, friction and hydrodynamics.",
    license="MIT LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="https://github.com/mattialabteam/elastica-python",
    author="Mattia-lab",
    download_url="https://github.com/mattialab/elastica-python/archive/master.zip",
    install_requires=['numpy','matplotlib','scipy']
)
