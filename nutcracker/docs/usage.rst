Usage of Nutcracker
===================

Follwing section trys to show how to execute *Nutcracker* functions. The denotation should be in accordance with :ref:`introduction`.

Intensities module
------------------

Fourier-Shell/Ring-Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    import nutcracker
    import numpy as np

    F1 = np.random.random((10,10)) #Fourier-Transform
    F2 = np.random.random((10,10)) #Fourier-Transforms

    FSC = nutcracker.intensities.fourier_shell_correlation(F1, F2)


Q-factor
^^^^^^^^

.. code::

    import nutcracker
    import numpy as np
    
    Fn = np.random.random((5,10,10)) #Set of Fourier-Transform 
    Q = nutcracker.intensities.q_factor(Fn)

Split image function
^^^^^^^^^^^^^^^^^^^^

.. code::

    import nutcracker
    import numpy as np
    
    F = np.random.random((8,8)) #Initial image

    F1, F2 = nutcracker.intensities.split_image(F, factor=2)

Quaternions module
------------------

Compare two sets of quaternions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

   import nutcracker
   import condor
   import numpy as np

   q1 = []
   for in range(10):
       q1.append(condor.utils.rotation.rand_quat())
   q1 = np.array(q_1)

   q2 = []
   for in range(10):
       q2.append(condor.utils.rotation.rand_quat())
   q2 = np.array(q2)

   output = quaternions.compare_two_sets_of_quaternions(q1, q2, n_samples=10, full_output=True, q1_is_extrinsic=True, q2_is_extrinsic=True)

Global quaternion rotation between two sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

   import nutcracker
   import condor
   import numpyas np

   q1 = []
   for in range(10):
       q1.append(condor.utils.rotation.rand_quat())
   q1 = np.array(q_1)

   q2 = []
   for in range(10):
       q2.append(condor.utils.rotation.rand_quat())
   q2 = np.array(q2)

   output = quaternions.global_quaternion_rotation_between_two_sets(q1, q2, full_output=True, q1_is_extrinsic=True, q2_is_extrinsic=True)

Real-space module
-----------------

Phase-Retrieval-Transfer-Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    import nutcracker
    import numpy as np

    img = np.random.random((8,10,10,10))
    sup = np.ones((8,10,10,10))

    PRTF_output = nutcracker.real_space.phase_retieval_transfer_function(img,sup,full_output=True)
    
    PRTF = PRTF_output['prtf_radial']

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
