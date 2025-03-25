============================
Reference API
============================

A thorough documentation of the MaDDG API.

Internally, MaDDG is divided into two core components: ``madlib`` and ``maddg``. 
``madlib`` contains the individual objects that are used to construct a simulation,  
such as satellites, maneuvers, sensors. ``maddg`` provides the framework that combines 
these objects into simulations and then deploys those simulations to compute nodes.

madlib
======

Satellites
----------

Satellites in MaDDG can be created from an initial set of 
position and velocity vectors, initial Keplerian elements, or 
even just the latitude and longitude for a geostationary 
object.

The :func:`~madlib._satellite.Satellite` class supports 
impulsive maneuvers, while the 
:func:`~madlib._satellite.ContinuousThrustSatellite` class 
can support continuous maneuvers.

.. currentmodule:: madlib._satellite

.. autosummary::
    :toctree: generated/

    Satellite
    ContinuousThrustSatellite

Sensors
-------

Currently, MaDDG only supports optical sensors. These sensors 
report the angular position of a target satellite (in Right Ascension 
and Declination) at scheduled times. Sensors can be ground- 
or space-based, and each sensor can be configured with 
unique noise profiles and observing conditions.

.. currentmodule:: madlib._sensor

.. autosummary::
    :toctree: generated/

    _Sensor
    _OpticalSensor
    GroundOpticalSensor
    SpaceOpticalSensor

.. currentmodule:: madlib._sensor_collection

Multiple sensors can be combined into a network, each providing 
its own measurements of the target satellite.

.. autosummary::
    :toctree: generated/

    SensorCollection

Observations
------------

.. currentmodule:: madlib._observation

Sensors' measurements of the target satellite's position 
are called observations. In most cases, we are actually 
interested in the target satellite's residuals, which 
are the difference between the object's measured position 
and its expected position at any given time.

.. autosummary::
    :toctree: generated/

    Observation
    ObservationResidual
    ObservationCollection

Maneuvers
---------

Satellites perform different types of maneuvers for a wide 
variety of reasons, from small burns to maintain an orbit to 
large burns that drastically alter the trajectory.

Maneuvers in MaDDG are divided into impulsive and continuous.

.. currentmodule:: madlib._maneuver

.. autosummary::
    :toctree: generated/

    ImpulsiveManeuver
    ContinuousManeuver

maddg
=====

Residual Calculation
--------------------

.. currentmodule:: maddg._residuals

The :func:`~maddg._residuals.calculate_residuals` method 
is a core simulation function. Given a network of sensors, 
a target satellite, a simulation duration and a start time, 
it will propagate the target and sensors through time and 
return a time series of the target's residuals in Right Ascension 
and Declination.

These residuals show the difference between the target's observed 
position and its expected position, making them a useful metric 
for detecting and characterizing satellite maneuvers.

.. autosummary::
    :toctree: generated/

    calculate_residuals

Launching Simulations
---------------------

MaDDG is designed to launch multiple simulations, and it 
can launch them in parallel if you have a multi-core CPU or 
access to a SLURM cluster.

.. currentmodule:: maddg._sim_launcher

The :func:`~maddg._sim_launcher.launcher` method is used to 
define a set of simulations. Users can define the satellites 
and sensors, assign the compute infrastructure and 
configure the simulation outputs.

.. autosummary::
    :toctree: generated/

    create_task_fn
    launcher