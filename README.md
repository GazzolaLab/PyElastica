<div align='center'>
<h1> PyElastica </h1>

[![Build_status][badge-travis]][link-travis] [![CI][badge-CI]][link-CI] [![Documentation Status][badge-docs-status]][link-docs-status] [![codecov][badge-codecov]][link-codecov] [![Downloads][badge-pepy-download-count]][link-pepy-download-count] [![Binder][badge-binder]][link-binder] [![Gitter][badge-gitter]][link-gitter]
 </div>
 
PyElastica is the python implementation of **Elastica**: an *open-source* project for simulating assemblies of slender, one-dimensional structures using Cosserat Rod theory.

[![gallery][link-readme-gallary]][link-project-website]

Visit [cosseratrods.org][link-project-website] for more information and learn about Elastica and Cosserat rod theory.

## How to Start 
[![PyPI version][badge-pypi]][link-pypi] [![Documentation Status][badge-docs-status]][link-docs-status]

PyElastica is compatible with Python 3.7 - 3.10.

~~~bash
$ pip install pyelastica 
~~~

Documentation of PyElastica is available [here][link-docs-website].

## Citation

We ask that any publications which use Elastica cite as following:

```
@misc{tekinalp2022pyelastica,
  title={PyElastica: A computational framework for Cosserat rod assemblies},
  author={Tekinalp, Arman and Kim, Seung Hyun and Parthasarathy, Tejaswin and Bhosale, Yashraj},
  journal={https://github.com/GazzolaLab/PyElastica},
  year={2022},
  publisher={GitHub}
}
```

<details>
  <summary><h4>References</h4></summary>
 
- Gazzola, Dudte, McCormick, Mahadevan, <strong>Forward and inverse problems in the mechanics of soft filaments</strong>, Royal Society Open Science, 2018. doi: [10.1098/rsos.171628](https://doi.org/10.1098/rsos.171628)
- Zhang, Chan, Parthasarathy, Gazzola, <strong>Modeling and simulation of complex dynamic musculoskeletal architectures</strong>, Nature Communications, 2019. doi: [10.1038/s41467-019-12759-5](https://doi.org/10.1038/s41467-019-12759-5)

</details>

## List of publications and submissions

- [Control-oriented modeling of bend propagation in an octopus arm](https://arxiv.org/abs/2110.07211) (UIUC, 2021)
- [A physics-informed, vision-based method to reconstruct all deformation modes in slender bodies](https://arxiv.org/abs/2109.08372) (UIUC, 2021) (IEEE ICRA 2022) [code](https://github.com/GazzolaLab/BR2-vision-based-smoothing)
- [Optimal control of a soft CyberOctopus arm](https://ieeexplore.ieee.org/document/9483284) (UIUC, 2021) (ACC 2021)
- [Elastica: A compliant mechanics environment for soft robotic control](https://ieeexplore.ieee.org/document/9369003) (UIUC, 2021) (IEEE RA-L 2021)
- [Controlling a CyberOctopus soft arm with muscle-like actuation](https://arxiv.org/abs/2010.03368) (UIUC, 2020)
- [Energy shaping control of a CyberOctopus soft arm](https://ieeexplore.ieee.org/document/9304408) (UIUC, 2020) (IEEE CDC 2020)

## Tutorials
[![Binder][badge-binder-tutorial]][link-binder]

We have created several Jupyter notebooks and Python scripts to help users get started with PyElastica. The Jupyter notebooks are available on Binder, allowing you to try out some of the tutorials without having to install PyElastica.

We have also included an example script for visualizing PyElastica simulations using POVray. This script is located in the examples folder ([`examples/visualization`](examples/visualization)).

## Contribution

If you would like to participate, please read our [contribution guideline](CONTRIBUTING.md)

PyElastica is developed by the [Gazzola Lab][link-lab-website] at the University of Illinois at Urbana-Champaign.

## Senior Developers ✨
_Names arranged alphabetically_
- Arman Tekinalp
- Chia-Hsien Shih (Cathy)
- Fan Kiat Chan
- Noel Naughton
- [Seung Hyun Kim](https://github.com/skim0119)
- Tejaswin Parthasarathy (Teja)
- [Yashraj Bhosale](https://github.com/bhosale2)

[//]: # (Collection of URLs.)

[link-readme-gallary]: https://github.com/skim0119/PyElastica/blob/assets_logo/assets/alpha_gallery.gif

[link-project-website]: https://cosseratrods.org
[link-lab-website]: http://mattia-lab.com/
[link-docs-website]: https://docs.cosseratrods.org/

[badge-pypi]: https://badge.fury.io/py/pyelastica.svg
[badge-travis]: https://travis-ci.com/GazzolaLab/PyElastica.svg?branch=master
[badge-CI]: https://github.com/GazzolaLab/PyElastica/workflows/CI/badge.svg
[badge-docs-status]: https://readthedocs.org/projects/pyelastica/badge/?version=latest
[badge-binder]: https://mybinder.org/badge_logo.svg
[badge-pepy-download-count]: https://pepy.tech/badge/pyelastica
[badge-codecov]: https://codecov.io/gh/GazzolaLab/PyElastica/branch/master/graph/badge.svg
[badge-gitter]: https://badges.gitter.im/PyElastica/community.svg
[link-pypi]: https://badge.fury.io/py/pyelastica
[link-travis]: https://travis-ci.com/github/GazzolaLab/PyElastica
[link-CI]: https://github.com/GazzolaLab/PyElastica/actions
[link-docs-status]: https://docs.cosseratrods.org/en/latest/?badge=latest
[link-pepy-download-count]: https://pepy.tech/project/pyelastica
[link-codecov]: https://codecov.io/gh/GazzolaLab/PyElastica

[badge-binder-tutorial]: https://img.shields.io/badge/Launch-PyElastica%20Tutorials-579ACA.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC
[link-binder]: https://mybinder.org/v2/gh/GazzolaLab/PyElastica/master?filepath=examples%2FBinder%2F0_PyElastica_Tutorials_Overview.ipynb
[link-gitter]: https://gitter.im/PyElastica/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
