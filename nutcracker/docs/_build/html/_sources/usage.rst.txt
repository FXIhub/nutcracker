Description of Nutcracker and how to use it
===========================================

The following section trys to give a basic understanding of the *Nutcracker* package. Futhermore some mathematical convetions are uesed. Vectors are denoted in bold letters. If just one componend of a vector is used, the related position in the vector is given as a footnote.  

Intensities module
------------------

The main features of this module are the Fourier-Shell-Correlation (3D), respectively the Fourier-Ring-Correlation (2D) and the Q-factor calculation. The side feature, the split image function, allows to split up one pattern or volume in two, to raise the opporunity to perform comparative image analysis on just one pattern/volume.

Fourier-Shell/Ring-Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fourier-Shell-Correlation (FSC) first introduced by Harauz and van Heel in 1986, is the 3 dimensional extention of the Fourier-Ring-Correlation (FRC). It basicly describes the similarity of two Fourier-Transforms :math:`F_{1}(\textbf{k})` and :math:`F_{2}(\textbf{k})`, with :math:`\textbf{k}` as spatial frequency vector. It starts at low spatial frequency with perfect correlation 1 and drops for increasing frequency. Usually the threshold for the resolution determination is :math:`\frac{1}{e}`. The purpose of this function is to calculate the resolution between two Fourier-Transforms.

.. math::
    FSC(\textbf{k}) = \frac{\sum_{k} F_{1}(\textbf{k}) F_{2}^{*}(\textbf{k})}{\sqrt{\sum_{k} \lvert F_{1}(\textbf{k}) \rvert^{2} \sum_{k} \lvert F_{2}(\textbf{k}) \rvert^{2}}}

Q-factor
^^^^^^^^

The Q-factor is calculated over a set of :math:`n` patterns and describes the ratio of the length of the Fourier-Transforms sum and the length of each Fourier-Transforms. The result is a 2D map which shows the clarity of the signal. This function is a fast and simple method to estimate the signal to noise ratio.

.. math::
    Q(\textbf{k}) = \frac{\lvert \sum_{n} F_{n}(\textbf{k}) \rvert}{\sum_{n} \lvert F_{n}(\textbf{k}) \rvert}

Split image function
^^^^^^^^^^^^^^^^^^^^

This function raises the opportunity to perform e.g. a FRC on a single pattern. The image or volume is splitted by creating superpixels in which single pixels are summed up in order or randomly according to the two resulting images. Thats conserve the number of scattered photons respectively the information of the actual input.

.. image:: ./images/split_image.png

Quaternions module
------------------

Quaternions are 4 dimensional complex number, with a scalar part :math:`w` and a vector part :math:`[x,y,z]`. The scalar part describes also the real part and the vector the imaginary part of the complex number. Additionally the three components of the imaginary part are multiplied by :math:`i`, :math:`j` and :math:`k`. Quaternions can be denoted as

.. math::
    q = w + xi + yj + zk
    \textbf{q} = [w,x,y,z].

The scalar or inner product in quaternion space is defined as

.. math::
    \bra{q_{1}} \ket{q_{2}} = \lvert q_{1} \rvert \lvert q_{2} \rvert \cos{\frac{\theta}{2}}.

This leads to an enclosed angle

.. math::
    \theta = 2 \arccos{\bra{q_{1}} \ket{q_{2}}} = 2 \arctan2{\lvert \textbf{q}_{r} \lvert, q_{r,0}}.

:math:`q_{r}` stands for the relative quaternion, where :math:`\textbf{q}_{r}` denotes the vector part and :math:`q_{r,0}` the scalar part.
The relative quaternion, meaning the rotation between two quaternions can be obtained by

.. math:: q_{r} = q_{1}^{-1} \cdot q_{2}

In this expression the :math:`\cdot` means the quaternion multiplikation.

Quaternions can be used to describe an intuitive way of rotations. Nonetheless one should take the ambiguity :math:`q = -q` into account.

The two following ways of comparing sets of quaternions originate from [Ekeberg et al., in progress].

Compare two sets of quaternions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function takes two sets of quaternions and pickes randomly quaternions within each set. By calculating now the relative quaternion between the two samples and the corresponding rotaion angle, the result is a distribution of the enclosed angle and shows how well the sets are correlating with each other. For more analysis one can look on the z-score or the percentage of the quaternions in a given standard deviation :math:`\sigma`.

Global quaternion rotation between two sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method gives the global rotation in form of a quaternion between the two sets. The result could than be used to realign the two sets or a corresponding model. At first, as in the method before, the relative quaternion is calculated, but now between each quaternion from the two sets. The mean of this relative quaternions lead to the global quaternion. To verify that the global roation is right, the enclosed angles between the global quaternion and the quaternions of one set are calculated. For a perfect correlaction between the two sets of quaternions the angular error is 0.
