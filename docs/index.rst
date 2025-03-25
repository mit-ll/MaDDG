.. admonition:: Join the Discussion

    Feel free to share ideas and ask questions over at `MaDDG's discussion page`_.

    .. _MaDDG's discussion page: https://github.com/mit-ll/MaDDG/discussions

=================================
Welcome to MaDDG's documentation!
=================================

MaDDG (**Ma**neuver **D**etection **D**ata **G**eneration) is a library for simulating 
high-fidelity observations of satellite trajectories with configurable maneuvers 
and custom sensor networks. MaDDG provides a simple interface for modeling complex 
observation scenarios. It allows you to create a satellite in any geocentric orbit, 
propagate its motion with a robust physical model, and track its position through 
optical sensors with customizable locations, observing limits, and noise parameters.

Through its use of `hydra-zen`_ and the `submitit plugin`_, MaDDG` can easily configure 
an array of simulation scenarios and distribute them in a SLURM cluster, empowering users 
to create large-scale, realistic datasets for training reliable maneuver detection and 
characterization models.

.. _hydra-zen: https://github.com/mit-ll-responsible-ai/hydra-zen
.. _submitit plugin: https://hydra.cc/docs/plugins/submitit_launcher/

Installation
============

MaDDG is available on PyPI:

.. code:: console

   $ pip install MaDDG

To install from source, clone the `MaDDG repository`_ and run the following command from 
its top-level directory:

.. _MaDDG repository: https://github.com/mit-ll/MaDDG

.. code:: console

   $ pip install -e .

If you want to modify the orbit propagation physics behind MaDDG, you
will likely need to edit the `AstroForge`_() library, as well. AstroForge is
an open-source astrodynamics library and a key requirement of MaDDG. See
the `AstroForge documentation`_ for installation instructions.

.. _AstroForge: https://github.com/mit-ll/AstroForge
.. _AstroForge documentation: https://astroforge.readthedocs.io/en/latest/

Documentation Contents
======================

.. toctree::
    :maxdepth: 2

    How-To Guides <how_to_guides>
    Reference <reference>