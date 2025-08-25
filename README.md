<div align='center'>
<h1> PyElastica </h1>

[![CI][badge-CI]][link-CI] [![Documentation Status][badge-docs-status]][link-docs-status] [![codecov][badge-codecov]][link-codecov] [![Downloads][badge-pepy-download-count]][link-pepy-download-count] [![DOI][badge-doi]][link-doi] [![Binder][badge-binder]][link-binder] [![Gitter][badge-gitter]][link-gitter]
 </div>

PyElastica is the python implementation of **Elastica**: an *open-source* project for simulating assemblies of slender, one-dimensional structures using Cosserat Rod theory.

[![gallery][link-readme-gallary]][link-project-website]

Visit [www.cosseratrods.org][link-project-website] for more information and learn about Elastica and Cosserat rod theory.

## How to Start
[![PyPI version][badge-pypi]][link-pypi] [![Documentation Status][badge-docs-status]][link-docs-status]

PyElastica is compatible with Python 3.10+.

```bash
$ pip install pyelastica
```

For plotting videos, ffmpeg is typically used.

Documentation of PyElastica is available [here][link-docs-website].

## Related Projects

- Cosserat rod with magnetic field: [magneto-pyelastica](https://github.com/armantekinalp/MagnetoPyElastica)
    - Simulate magnetic Cosserat rods interacting with external magnetic environments.
    - `pip install magneto_pyelastica`
- gymnasium environment for learning and control: [gym-softrobot](https://github.com/skim0119/gym-softrobot)
- BR2 FREE pneumatic actuator simulation: [BR2-simulator](https://github.com/skim0119/BR2-simulator)
- Blender visualization: [Blender-Soft-Rod](https://github.com/GazzolaLab/Blender-Soft-Rod)
- Rhino3D plugin for visualizaion: [plugin](https://github.com/skim0119/PyElastica-to-Rhino)
- Elastica web-interface: [PyElastica-Interactive](https://github.com/GazzolaLab/PyElastica-Interactive)

## Citation

We ask that any publications which use Elastica cite as following:

```
@software{arman_tekinalp_2024_10883271,
  author       = {Arman Tekinalp and
                  Seung Hyun Kim and
                  Yashraj Bhosale and
                  Tejaswin Parthasarathy and
                  Noel Naughton and
                  Ali Albazroun and
                  Rahul Joon and
                  Songyuan Cui and
                  Ilia Nasiriziba and
                  Maximilian Stölzle and
                  Chia-Hsien (Cathy) Shih and
                  Mattia Gazzola},
  title        = {GazzolaLab/PyElastica},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10883271},
  url          = {https://doi.org/10.5281/zenodo.10883271}
}
```

<details>
  <summary><h4>References</h4></summary>

- Gazzola, Dudte, McCormick, Mahadevan, <strong>Forward and inverse problems in the mechanics of soft filaments</strong>, Royal Society Open Science, 2018. doi: [10.1098/rsos.171628](https://doi.org/10.1098/rsos.171628)
- Zhang, Chan, Parthasarathy, Gazzola, <strong>Modeling and simulation of complex dynamic musculoskeletal architectures</strong>, Nature Communications, 2019. doi: [10.1038/s41467-019-12759-5](https://doi.org/10.1038/s41467-019-12759-5)

</details>

## List of publications and submissions
- [Simulations and experiments with assemblies of fiber-reinforced soft actuators](https://arxiv.org/abs/2507.10121) (UIUC 2025)
- [Soft, slender and active structures in fluids: embedding Cosserat rods in vortex methods](https://doi.org/10.48550/arXiv.2401.09506) (UIUC 2024)
- [Neural models and algorithms for sensorimotor control of an octopus arm](https://doi.org/10.48550/arXiv.2402.01074)(UIUC 2024)
- [On the mechanical origins of waving, coiling and skewing in Arabidopsis thaliana roots](https://www.pnas.org/doi/10.1073/pnas.2312761121) (Tel Aviv University, UIUC 2024) (PNAS)
- [Topology, dynamics, and control of an octopus-analog muscular hydrostat](https://arxiv.org/abs/2304.08413) (UIUC, 2023)
- [Hierarchical control and learning of a foraging CyberOctopus](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300088) (UIUC, 2023) (Advanced Intelligent Systems)
- [Energy-shaping control of a muscular octopus arm moving in three dimensions](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2022.0593) (UIUC, 2023) (Proceedings of the Royal Society A 2023)
- [A sensory feedback control law for octopus arm movements](https://ieeexplore.ieee.org/abstract/document/9993021/) (UIUC, 2022) (IEEE CDC 2022)
- [Control-oriented modeling of bend propagation in an octopus arm](https://ieeexplore.ieee.org/abstract/document/9867689/) (UIUC, 2021) (IEEE ACC 2022)
- [A physics-informed, vision-based method to reconstruct all deformation modes in slender bodies](https://arxiv.org/abs/2109.08372) (UIUC, 2021) (IEEE ICRA 2022) [code](https://github.com/GazzolaLab/BR2-vision-based-smoothing)
- [Optimal control of a soft CyberOctopus arm](https://ieeexplore.ieee.org/document/9483284) (UIUC, 2021) (IEEE ACC 2021)
- [Elastica: A compliant mechanics environment for soft robotic control](https://ieeexplore.ieee.org/document/9369003) (UIUC, 2021) (IEEE RA-L 2021)
- [Controlling a CyberOctopus soft arm with muscle-like actuation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9683318) (UIUC, 2020) (IEEE CDC 2021)
- [Energy shaping control of a CyberOctopus soft arm](https://ieeexplore.ieee.org/document/9304408) (UIUC, 2020) (IEEE CDC 2020)

## Tutorials
[![Binder][badge-binder-tutorial]][link-binder]

We have created several Jupyter notebooks and Python scripts to help users get started with PyElastica. The Jupyter notebooks are available on Binder, allowing you to try out some of the tutorials without having to install PyElastica.

We have also included an example script for visualizing PyElastica simulations using POVray. This script is located in the examples folder ([`examples/Visualization`](examples/Visualization)).

## Contribution

If you would like to participate, please read our [contribution guideline](CONTRIBUTING.md). Private development branches are moved to [elastica-python](https://github.com/GazzolaLab/elastica-python) repository; access is limited to the core developers, collaborators, and maintainers.

PyElastica is developed by the [Gazzola Lab][link-lab-website] at the University of Illinois Urbana-Champaign.

## Senior Developers ✨
_Names arranged alphabetically_
- Ali Albazroun
- Arman Tekinalp
- Chia-Hsien Shih (Cathy)
- Fan Kiat Chan
- Ilia Nasiriziba
- Noel Naughton
- [Seung Hyun Kim](https://github.com/skim0119)
- Songyuan Cui
- Tejaswin Parthasarathy (Teja)
- Xiaotian Zhang
- [Yashraj Bhosale](https://github.com/bhosale2)

[//]: # (Collection of URLs.)

[link-readme-gallary]: https://github.com/skim0119/PyElastica/blob/assets_logo/assets/alpha_gallery.gif

[link-project-website]: https://cosseratrods.org
[link-lab-website]: http://mattia-lab.com/
[link-docs-website]: https://docs.cosseratrods.org/

[badge-pypi]: https://badge.fury.io/py/pyelastica.svg
[badge-CI]: https://github.com/GazzolaLab/PyElastica/workflows/CI/badge.svg
[badge-docs-status]: https://readthedocs.org/projects/pyelastica/badge/?version=latest
[badge-binder]: https://mybinder.org/badge_logo.svg
[badge-pepy-download-count]: https://pepy.tech/badge/pyelastica
[badge-codecov]: https://codecov.io/gh/GazzolaLab/PyElastica/branch/master/graph/badge.svg
[badge-gitter]: https://badges.gitter.im/PyElastica/community.svg
[badge-doi]: https://zenodo.org/badge/254172891.svg
[link-pypi]: https://badge.fury.io/py/pyelastica
[link-CI]: https://github.com/GazzolaLab/PyElastica/actions
[link-docs-status]: https://docs.cosseratrods.org/en/latest/?badge=latest
[link-pepy-download-count]: https://pepy.tech/project/pyelastica
[link-codecov]: https://codecov.io/gh/GazzolaLab/PyElastica

[badge-binder-tutorial]: https://img.shields.io/badge/Launch-PyElastica%20Tutorials-579ACA.svg?logo=data:image/png;base64,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
[link-binder]: https://mybinder.org/v2/gh/GazzolaLab/PyElastica/master?filepath=examples%2FBinder%2F0_PyElastica_Tutorials_Overview.ipynb
[link-gitter]: https://gitter.im/PyElastica/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[link-doi]: https://zenodo.org/badge/latestdoi/254172891
