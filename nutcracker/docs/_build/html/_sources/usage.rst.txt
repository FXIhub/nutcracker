Usage of Nutcracker anf package description
===========================================

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
