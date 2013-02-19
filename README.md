PyGuide
=======

Python program for analysing planar X-ray waveguides.

 I kludged this together as a part of a 3rd year physics research project at Monash University (they are really awesome at getting undergrads doing real research!) with Daniele Pelliccia (http://monash.edu/science/about/schools/physics/people/research/pelliccia.html) and David Paganin (http://monash.edu/science/about/schools/physics/people/academic/paganin.html).

If you've got any questions or comments about this (or if you're trying to use it and it's completley broken!), please do email me at kjtsa1@student.monash.edu! It wasn't really intended to be something generally useful when I wrote it, but I figured I may as well throw it on Github anyway in case I was wrong. For a discussion on the physics behind this, look at ``characterisation_of_an_x-ray_waveguide.pdf`` in the repo.

All the heavy lifting is done by the most excelent NumPy, SciPy and matplotlib libraries (http://www.numpy.org/, http://www.scipy.org/ and http://matplotlib.org/)

Usage
-------

PyGuide is a command-line program that accepts inputs predominantly from the command line and outputs plots as image files and tables as comma-separated text files.

All functions of PyGuide can accept the following parameters-

``-v``, ``--verbose``
Display calculation information as the program is running

``-wg [wg-name]``, ``--waveguide [wg-name]``
``-wgf [wg-file]``, ``--waveguidefile [wg-file]``
These options determine the characteristics of the waveguide that is being analysed. If the ``--waveguidefile`` flag is specified, then the waveguide characteristics are read from the specified file. Otherwise, if the ``--waveguide`` flag is specified, the waveguide is read from a file named ``[wg-name].xml`` from the ``waveguides`` folder. The details of this waveguide format are provided later in this document.

``-o [filename | dirname]``, ``--output [filename | dirname]``
File in which to store the output of this program. If multiple files are generated, this name will be taken as a directory name to create and store the output files in.

Finding Guided Modes
--------------------

PyGuide can calculate the guided modes of your waveguide by invoking it as ``python PyGuide [general-options] modesolver [output-type] [modesolver-options]``. The ``[output-type]`` flag is mandatory and specifies what kind of output will be generated.

``-kx``, ``--wavevectors``
Produces a CSV table containing the wavevector value and grazing angle of each of the guided modes in the waveguide

``-ip``, ``--intensityplot``
Produces plots of the square modulus of each guided mode wavefunction as individual image files. The longitudinal distance at which to calculate this is specified by the ``--distances`` flag

``-wfp``, ``--wavefunctionplot``
Produces plots of the entrance surface wavefunction of each guided mode as individual image files. 

``-pp``, ``--poyntingplot``
Produces plots of the Poynting vector at the entrance surface as a function of transverse distance of each guided mode as individual image files. 

``-ap``, ``--argandplot``
Produces plots of the entrance surface wavefunction of each guided mode on an argand diagram as individual image files. It looks cool, but I never understood what this meant physically.

The following options specify aspects of the behaviour of this mode calculation

``-m [int list]``, ``--modes [int list]``
A list of numbers representing which guided modes should have plots generated for them (ignored for the ``--wavevectors`` table). Don't specify this if you want to see plots of all modes; the syntax of it looks like ``--modes 3 4 5``.

``-d [distance]``, ``--distance [distance]``
The longitudinal distance (in metres) at which to generate the relevant plots; this is only used for the ``--poyntingplot`` plots. Exponential notation works (e.g. ``--distance 5e-3`` is 5mm).

``-l [wavelength]``, ``--wavelength [wavelength]``.
The wavelength of light for which to calculate the guided modes (applies to all output types). This is specified in metres and again exponential notation is OK.
 
More to come!
-------------


Waveguide File Format
---------------------

The waveguide file format is pretty self-documenting; look in the ``waveguides`` folder for some examples. Some things to note:

* the ``type`` parameter of the opening ``<waveguide>`` tag was supposed to describe the geometry of the waveguide; I only implemented planar waveguides, so this value actually is ignored
* The ``<length>`` tag is the length of the waveguide in metres
* The ``<slabgap>`` tag is the width of the core material in metres
* The ``<coreindex>`` tag is the complex refractive index of the core material
* The ``<claddingindex>`` tag contains a space seperated table consisting of three columns: wavelength (in nm), real part of refractive index, and complex part of refractive index. This table describes the refractive index of the cladding material as a function of wavelength. This table is interpolated to yield the refractive index at an arbitrary wavelength.
* Note that I just fit a straight line to the log of the refractive index and use it for the interpolation. This is no good if you have an absorption edge; you could do something different by changing ``PlanarWaveguideInterp.__init__()`` and ``PlanarWaveguideInterp.cladding_index()`` in ``waveguides.py``.