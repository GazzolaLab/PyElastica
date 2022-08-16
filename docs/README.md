# Elastica Documentation

We use [`Sphinx`](https://www.sphinx-doc.org/en/master/) and [`ReadtheDocs`](https://readthedocs.org/) to organize our [documentation page](https://docs.cosseratrods.org/en/latest/).

In addition, we utilize the following extensions to enhance the documentation :coffee:
- `numpydoc`: We favor [numpy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html) for API documentation.
- `myst_parser`: We like to write documentation and guidelines in `markdown` format.

## Build documentation

The `sphinx` is already initialized in `docs` directory. 
In order to build the documentation, you will need additional 
packages listed in extra dependencies.

```bash
poetry install -E docs
cd docs
make clean
make html
```

Once the documentation building is done, open `docs/_build/html/index.html` to view.

Use `make help` for other options.

# Contribution

The documentation-related commits will be collected in the branch `doc_**` separate from `master` branch, and merged into `master` collectively. Ideally, the updates in the documentation branch will seek upcoming version update (i.e. `update-**` branch) and merged shortly after the version release. If an update is critical and urgent, create PR directly to `master` branch.

## User Guide

User guidelines and tutorials are written in `.rst` or `.md` format.
These files will be managed in `docs` directory.

> In the future, a separate `asset` branch will be created to keep images and other binary files.

## API documentation

The docstring for function or modules are automatically parsed using `sphinx`+`numpydoc`.
Any inline function description, such as 

```py
""" This is the form of a docstring.

... description

Attributes
----------
x : type
    x description
y : type
    y description

"""
```

will be parsed and displayed in API documentation. See `numpydoc` for more details.

## ReadtheDocs

`ReadtheDocs` runs `sphinx` internally and maintain the documentation website. We will always activate the `stable` and `latest` version, and few past-documentations will also be available for the support.

@nmnaughton and @skim449 has access to the `ReadtheDocs` account.
