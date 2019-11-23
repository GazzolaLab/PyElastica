from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = "0.0.0"

setup(
    name="elastica",
    packages=['elastica'],
    package_dir={'elastica': './elastica'},
    version=version,
    description="Rod simulation",
    license="MIT LICENSE",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="https://github.com/armantekinalp/elastica-python",
    author="Arman Tekinalp",
    download_url="https://github.com/armantekinalp/elastica-python/archive/master.zip",
    install_requires=['numpy','matplotlib','scipy','jupyterlab']
)
