Usage of Nutcracker
===================

Follwing section trys to show how to execute *Nutcracker* functions. The denotation should be in accordance with :ref:`_introduction`.

Intensities module
------------------

Fourier-Shell/Ring-Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::
    .. highlight:: python

    import nutcracker
    import numpy as np

    F1 = np.random.random((10,10)) #Fourier-Transform
    F2 = np.random.random((10,10)) #Fourier-Transforms

    FSC = nutcracker.intensities.fourier_shell_correlation(F1, F2)


Q-factor
^^^^^^^^

::
    .. highlight:: python

    import nutcracker
    import numpy as np
    
    Fn = np.random.random((5,10,10)) #Set of Fourier-Transform 
    Q = nutcracker.intensities.q_factor(Fn)

Split image function
^^^^^^^^^^^^^^^^^^^^

Quaternions module
------------------

Compare two sets of quaternions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Global quaternion rotation between two sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real-space module
-----------------

Submodules
----------

Rotate
^^^^^^

Shift
^^^^^

Plot-analysis
^^^^^^^^^^^^^

Error matrix multiprocessed
^^^^^^^^^^^^^^^^^^^^^^^^^^^
