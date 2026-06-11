Pipeline Configs
================

This repository stores the pipeline configs that are used for
the various [SWIFT](http://swift.dur.ac.uk) simulations
using the [pipeline](http://github.com/swiftsim/pipeline).

The only required setup to use the configs is to format the
observational data, which is performed using the submodule:

```bash
git submodule update --init --recursive
cd observational_data
./convert.py [--nproc <NUMBER OF PARALLEL PROCESSES>]
cd ..
```

That should be everything you need to get going, as well as
the documentation available in the ``pipeline`` repository.

Prior to June 2026 there were config files for ``eagle-xl``
and ``colibre-zooms``, but these have bee removed as they 
were unmaintained.

Running on large simulations
-----------------------------

Some scripts may not be feasible to run on very large simulations due to
memory or runtime constraints. In these cases it may be necessary to disable
individual scripts in ``config.yml`` by removing or commenting out the relevant
entries.

The autoplotting phase can also be very slow for large simulations. Scatter
plots render one point per galaxy, and with hundreds of thousands of objects
matplotlib's ``scatter`` can take minutes per plot. This means the autoplotter
phase alone can take several hours on the largest simulations.
