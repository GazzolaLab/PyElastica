# Contributing to PyElastica

Thanks for taking the time to contribute!

The following is a set of guidelines for contributing to PyElastica. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[TLTR! I need three-line summary!!](#three-line-summary)

[Before I get started?](#before-i-get-started)
  * [Installation and packages](#installation-and-packages)
  * [Project workflow](#project-workflow)

[How can I contribute?](#how-can-i-contribute)
  * [Reporting bugs](#reporting-bugs)
  * [Suggesting enhancements](#suggesting-enhancements)
  * [Your first code contribution](#your-first-code-contribution)
  * [Side projects](#side-projects)
  * [Pull requests](#pull-requests)
  * [Have questions about the source code](#have-questions-about-the-source-code)

[Styleguides](#styleguides)
  * [Git commit messages](#git-commit-messages)
  * [Formatting and styleguide](#formatting-and-styleguide)
  * [Documentation styleguide](#documentation-styleguide)

## Three-line summary

1. I found bugs! - Please **ensure the bug was not already reported** on issues. If you cannot find an open issue addressing the problem, **open a new one**. Be sure to include clear **title and description** and appropriate **labels**. To speed up the process, we recommend including **how to reproduce behavior** along with **expected behavior**.
2. I'm struggling in using PyElastica for my project! - Please open the issue with the label `help wanted` and explain the details of the problem and issue. We would gladly reach you to assist.
3. How can I contribute!? - If you already wrote the patch, you can open a `pull request` with the changes to the `update_<version>` branch. Ensure the code passes the [formatting and styles](#formatting-and-styleguide). The PR should clearly describe the problem and solution. Include all relevant issue numbers if applicable.

## Before I get started

### Setup development environment

Below are steps of how to setup development environment. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.8.0.

```bash
conda create --name pyelastica-dev
conda activate pyelastica-dev
conda install python==3.8
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```

If you are planning to contribute to the examples,
extra dependencies can be installed using `poetry install -E examples`.

If you are planning to contribute on documentation, extra dependencies can be installed
using `poetry install -E docs`.
The detail instruction is included
[here](https://github.com/GazzolaLab/PyElastica/blob/master/docs/README.md).

4. Now your working environment is set!

### Project workflow

We will create an `update_<version>` branch every couple of months, and will release the branch once all related issues are resolved.
Every patch will be merged into the update branch only to organize major changes.
Each version will have an associated collection of issues as `Milestones`.

If you have a patch, make a `pull request` to the `update_<version>` branch.
We recommend making the **branch name start with related issue number** (ex. `1_<branch name>`).

## How can I contribute?

### Reporting bugs

*Following these guidelines helps maintainers and the community to understand the problem.*

<!-- We provide [bug report template][link-issue-bug-report] to guide you resolving issues faster.) -->

* Please check if a **related bug is not already issued**.
* Use a **clear and descriptive title** for the issue to identify the problem. Also, include appropriate **labels**.
* Reporting **how to reproduce a bug** and **details of the problem (constraint, connection, etc.)** could help developers to resolve the issue. Including code is also a good idea.
* **Explain the expected behavior and why.**
* Include **additional details** as much as possible.
	* Including **screenshots or animated gifs** also helps.
	* Which **version of PyElastica and dependencies** was used.
	* If relevant, what **version and OS** was used.
* Link any related issues using `#<issue number>`.

> **Note:** If you find a **Closed** issue that seems like the same thing that you're experiencing, please open a new issue and include a link to the original (closed) issue in the description.

### Suggesting enhancements

This section guides you through submitting an enhancement suggestion, including new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion.

<!--  We provide [the template][link-issue-feature-request] to guide you resolving issues faster. -->

* Use a **clear and descriptive title** for the issue to identify the suggestion. Also include appropriate **labels:enhancements**.
* If the problem is related to **performance or memory**, please include a performance profiling that you are reporting.
* Please include as many **details** as possible. If you already have a suggestion patch, please include the link to the changes.
	* Which **version of PyElastica and dependencies** was used.
	* If relevant, what **version and OS** was used.
* Include a description of the **current behavior** and **explain expected behavior**.
* Explain **why this enhancement would be useful**.

### Your first code contribution

If you are unsure regarding how to start contributing to PyElastica, you can look through `good first issue` and `help wanted` labels.
Also, you might find `TODO` or `FIXME` marks in various places in the code with brief descriptions.

* [Beginner issues][beginner] - issues that should only require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit more involved and might need some discussion. You might need more in-depth understanding of the theory and implementation to resolve issues with this tag.

Once you find the issue that you are interested in, please leave a comment that you would like to resolve the issue.
If you leave some questions in the comment, we will provide more detailed descriptions.

You are also welcomed to help us pushing this project further.
Please don't hesitate improving [documentation](https://github.com/GazzolaLab/PyElastica/tree/master/docs) or code-coverage.

### Side projects

We also have many related projects in separate repositories that utilize the PyElastica as a core library.
Since the package is often used for research purpose, many on-going projects are typically not published.
If you are interested in hearing more, please contact one of our the maintainer.

### Pull requests

Please follow these steps to have your contribution considered by the maintainers:

1. Follow the [styleguides](#styleguides)
2. If you add a new dependency, add it to the `pyproject.toml` and then run the following line from the top directory:

    `
   make install_with_new_dependency
   `

	This will update `poetry.lock` to ensure version control. Don't forget to commit `.lock` and `.toml` files for Poetry in this case!
3. Before you submit your pull request run [pytests](https://pypi.org/project/pytest/) and make sure that all tests pass.

    In order to run pytest, run the following line from the top directory:

    `
    make test
    `

4. After you submit your pull request, verify that all status checks are passing <details><summary>What if the status checks are failing?</summary>If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the pull request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.</details>

The reviewer(s) may ask you to complete additional tests or changes before your pull request can be accepted.

### Have questions about the source code?

Ask any question about **how to use PyElastica and detail implementation** in the **issue with label:question**.

## Styleguides

### Git commit messages

* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* When only changing documentation, include `[ci skip]` in the commit title

### Formatting and styleguide

We use [flake8](https://pypi.org/project/flake8/) and [Black](https://pypi.org/project/black/) for python style.

In order to format the code:

`make formatting`

> **Note:** Format/refactoring patches that are not anything substantial to the context or functionality will likely be rejected.

### Documentation styleguide

We follow [NumPy documentation guidelines][numpydoc-guideline] for PyElastica documentation.
If you are interested in contributing or modifying the documentation, please refer to the [docs-readme][docs-readme] file.

[beginner]: https://github.com/GazzolaLab/PyElastica/labels/good%20first%20issue
[help-wanted]: https://github.com/GazzolaLab/PyElastica/labels/help%20wanted

[numpydoc-guideline]: https://numpydoc.readthedocs.io/en/latest/format.html
[docs-readme]: https://github.com/GazzolaLab/PyElastica/blob/master/docs/README.md
