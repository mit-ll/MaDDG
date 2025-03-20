# MaDDG

<p align="center">A library for simulating high-fidelity observations of satellite trajectories with configurable maneuvers and custom sensor networks.</p>

`MaDDG` provides a simple interface for modeling complex observation scenarios. It allows you to create a satellite in any geocentric orbit, propagate its motion with a robust physical model, and track its position through optical sensors with customizable locations, observing limits, and noise parameters.

Through its use of [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) and the [submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher/), `MaDDG` can easily configure an array of simulation scenarios and distribute them in a SLURM cluster, empowering users to create large-scale, realistic datasets for training reliable maneuver detection and characterization models.


## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/
