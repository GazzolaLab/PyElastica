# PyElastica 
[![Build_status](https://travis-ci.com/GazzolaLab/PyElastica.svg?branch=master)](https://travis-ci.com/github/GazzolaLab/PyElastica)  [![Documentation Status](https://readthedocs.org/projects/pyelastica/badge/?version=latest)](https://docs.cosseratrods.org/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/GazzolaLab/PyElastica/branch/master/graph/badge.svg)](https://codecov.io/gh/GazzolaLab/PyElastica)  [![Downloads](https://pepy.tech/badge/pyelastica)](https://pepy.tech/project/pyelastica) [![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GazzolaLab/PyElastica/master?filepath=examples%2FBinder%2F0_PyElastica_Tutorials_Overview.ipynb)

PyElastica is the python implementation of **Elastica**, which is a *free* and *open-source* software project for the simulation of assemblies of slender, one-dimensional structures using Cosserat Rod theory. More information about Elastica and Cosserat rod theory is available at the Elastica [project website](https://cosseratrods.org)

## New this Release
This release of PyElastica uses the Python package `numba` to enable just in time compilation leading to a ~8x speedup over the previous version. Numba is not required to run PyElastica and if numba is not installed, PyElastica will defualt to the non-numba implementation. As such, if you wish to take advantage of the speed-up afforded by numba, please be sure to install it separately.

Future releases of PyElastica will require numba and we will no longer be maintaining the non-numba code beyond this release. 

We have also included an example script for visualizing PyElastica simulations using POVray. This script is located in the examples folder (`examples/visualization`).

## Installation 
[![PyPI version](https://badge.fury.io/py/PyElastica.svg)](https://badge.fury.io/py/PyElastica)

PyElastica is compatible with Python 3.5 - 3.8. The easiest way to install PyElastica is with PIP. 

~~~bash
$ pip install pyelastica 
~~~

To provide the best performance, this package requires the numba package. If for some reason you can not use numba you can still download PyElastica from here and run it locally without numba installed. 

Previous PyElastica releases are available in the branches. 

## Documentation
[![Documentation Status](https://readthedocs.org/projects/pyelastica/badge/?version=latest)](https://docs.cosseratrods.org/en/latest/?badge=latest)

Documentation of PyElastica is available at [docs.cosseratrods.org](https://docs.cosseratrods.org/)

PyElastica is developed by the [Gazzola Lab](http://mattia-lab.com/) at the University of Illinois at Urbana-Champaign. 

## Tutorials
[![Binder](https://img.shields.io/badge/Launch-PyElastica%20Tutorials-579ACA.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC)](https://mybinder.org/v2/gh/GazzolaLab/PyElastica/master?filepath=examples%2FBinder%2F0_PyElastica_Tutorials_Overview.ipynb)

We have created several Jupyter notebooks and Python scripts to help get users started with using PyElastica. The Jupyter notebooks are available on Binder, allowing you to try out some of the tutorials without having to install PyElastica.   

## Citation
We ask that any publications which use Elastica cite the following papers:  

Zhang, Chan, Parthasarathy, Gazzola, <strong>Modeling and simulation of complex dynamic musculoskeletal architectures</strong>, Nature Communications, 2019. doi: [10.1038/s41467-019-12759-5](https://doi.org/10.1038/s41467-019-12759-5)

Gazzola, Dudte, McCormick, Mahadevan, <strong>Forward and inverse problems in the mechanics of soft filaments</strong>, Royal Society Open Science, 2018. doi: [10.1098/rsos.171628](https://doi.org/10.1098/rsos.171628)
```
@article{gazzola2018forward,
  title={Forward and inverse problems in the mechanics of soft filaments},
  author={Gazzola, M and Dudte, LH and McCormick, AG and Mahadevan, L},
  journal={Royal Society open science},
  volume={5},
  number={6},
  pages={171628},
  year={2018},
  publisher={The Royal Society Publishing},
  doi = {10.1098/rsos.171628},
  url = {https://doi.org/10.1098/rsos.171628},
}
```

```
@article{zhang2019modeling,
  title={Modeling and simulation of complex dynamic musculoskeletal architectures},
  author={Zhang, X and Chan, FK and Parthasarathy, T and Gazzola, M},
  journal={Nature Communications},
  volume={10},
  number={1},
  pages={1--12},
  year={2019},
  publisher={Nature Publishing Group},
  doi = {10.1038/s41467-019-12759-5},
  url = {https://doi.org/10.1038/s41467-019-12759-5},
}
```
