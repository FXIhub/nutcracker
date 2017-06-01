Installation
============

1) Dependencies
---------------
Following packages are requiered to run Nutcracker:

    - numpy
    - scipy
    - h5py
    - `spimage <https://github.com/FXIhub/libspimage>`_
    - `condor <https://github.com/FXIhub/condor>`_
    - `mulpro <https://github.com/mhantke/mulpro>`_

To install numpy, scipy and h5py use the pip:

.. code::

    $ pip install numpy scipy h5py --user

To get spimage, condor and mulpro use the git clone command:

.. code::

    $ git clone <SSH-key> #or
    $ git clone <HTTPS>

For more information about installing the packages visit the corresponding webpage.

2) Installing Nutcracker
------------------------
To install nutcracker one proceed as for condor or mulpro.

.. code::

    $ git clone git@github.com:FXIhub/nutcracker.git # for SSH-key
    $ git clone https://github.com/FXIhub/nutcracker.git # for HTTPS
    $ cd nutcracker
    $ python setup.py install --user

Congratulations! You are now ready to use Nutcracker.
