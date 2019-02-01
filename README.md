# Sample code for shape optimisation in Firedrake

The code presented here requires the [Firedrake](https://firedrakeproject.org) finite element library.
On Mac or Linux, installing Firedrake is as easy as 

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

(see [here](https://firedrakeproject.org/download.html) for details).
On Windows 10, Firedrake can be used using the Windows Subsystem for Linux. Detailed Instructions can be found [here](https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux). Firedrake is unlikely to work on older versions of Windows.

Once Firedrake has been set up, it needs to be activated using 

    source /path/to/firedrake/bin/activate

The code can then be run using

    python3 pipe2dsolve.py

To run the full shape optimisation code, execute

    python3 pipe2dopt.py
