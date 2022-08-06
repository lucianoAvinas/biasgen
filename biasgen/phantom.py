"""
(Note: reduced the number of phantoms modeled)

Creating Shepp-Logan like phantoms in 2D and 3D storing them as DEN
@author: Vojtěch Kulvait
(c) 2020

Original inspiration for the code is
https://gist.github.com/blink1073/6e417c726b0dd03d5ea0
Code was aparently inspirred by the contribution of Matthias Christian Schabel to Mathworks
Matthias Schabel (2020). 3D Shepp-Logan phantom
https://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom

3D phantoms nxnxn and n is a parameter of the computation
Specification is however given on continuous grid of [-1,1] x [-1, 1] or [-1,1] x [-1, 1] x [-1, 1]
Continuous grids are mapped to voxel grids

In this file the following set of phantoms is predefined:

SheppLogan2D phantom
====================
Phantom from doi.org/10.1109/TNS.1974.6499235

    A           a           b           x0          y0          phi

  2.00000     0.69000     0.92000     0.00000     0.00000     0.00000
 -0.98000     0.66240     0.87400     0.00000    -0.01840     0.00000
 -0.02000     0.11000     0.31000     0.22000     0.00000   -18.00000
 -0.02000     0.16000     0.41000    -0.22000     0.00000    18.00000
  0.01000     0.21000     0.25000     0.00000     0.35000     0.00000
  0.01000     0.04600     0.04600     0.00000     0.10000     0.00000
  0.01000     0.04600     0.04600     0.00000    -0.10000     0.00000
  0.01000     0.04600     0.02300    -0.08000    -0.60500     0.00000
  0.01000     0.02300     0.02300     0.00000    -0.60500     0.00000
  0.01000     0.02300     0.04600     0.06000    -0.60500     0.00000


Toft2D phantom
==============
Also mod_shepp_logan image-2.12.0, taken from page 201, section B.3 of Peter Toft Ph.D. thesis 
https://petertoft.dk/PhD/PeterToft_PhD_thesis_5.pdf

    A           a           b           x0          y0          phi

  1.00000     0.69000     0.92000     0.00000     0.00000     0.00000
 -0.80000     0.66240     0.87400     0.00000    -0.01840     0.00000
 -0.20000     0.11000     0.31000     0.22000     0.00000   -18.00000
 -0.20000     0.16000     0.41000    -0.22000     0.00000    18.00000
  0.10000     0.21000     0.25000     0.00000     0.35000     0.00000
  0.10000     0.04600     0.04600     0.00000     0.10000     0.00000
  0.10000     0.04600     0.04600     0.00000    -0.10000     0.00000
  0.10000     0.04600     0.02300    -0.08000    -0.60500     0.00000
  0.10000     0.02300     0.02300     0.00000    -0.60600     0.00000
  0.10000     0.02300     0.04600     0.06000    -0.60500     0.00000


ToftSchabel3D phantom
=================
Specification of 10 ellipsoids given in Matthias Schabel (2020). 3D Shepp-Logan phantom contribution inspirred by
Toft2D Matthias Schabel (2020). 3D Shepp-Logan phantom
https://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom

    A           a           b           c           x0          y0          z0          phi         theta       psi

  1.00000     0.69000     0.92000     0.81000     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000
 -0.80000     0.66240     0.87400     0.78000     0.00000    -0.01840     0.00000     0.00000     0.00000     0.00000
 -0.20000     0.11000     0.31000     0.22000     0.22000     0.00000     0.00000   -18.00000     0.00000    10.00000
 -0.20000     0.16000     0.41000     0.28000    -0.22000     0.00000     0.00000    18.00000     0.00000    10.00000
  0.10000     0.21000     0.25000     0.41000     0.00000     0.35000    -0.15000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.04600     0.05000     0.00000     0.10000     0.25000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.04600     0.05000     0.00000    -0.10000     0.25000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.02300     0.05000    -0.08000    -0.60500     0.00000     0.00000     0.00000     0.00000
  0.10000     0.02300     0.02300     0.20000     0.00000    -0.60600     0.00000     0.00000     0.00000     0.00000
  0.10000     0.02300     0.04600     0.20000     0.06000    -0.60500     0.00000     0.00000     0.00000     0.00000


ToftSchabelKulvait3D phantom
===========================
ToftSchabel3D with all zero theta psi and z0 so that center slice is Toft2D

     A           a           b           c           x0          y0          z0          phi         theta       psi

  1.00000     0.69000     0.92000     0.81000     0.00000     0.00000     0.00000     0.00000     0.00000     0.00000
 -0.80000     0.66240     0.87400     0.78000     0.00000    -0.01840     0.00000     0.00000     0.00000     0.00000
 -0.20000     0.11000     0.31000     0.22000     0.22000     0.00000     0.00000   -18.00000     0.00000     0.00000
 -0.20000     0.16000     0.41000     0.28000    -0.22000     0.00000     0.00000    18.00000     0.00000     0.00000
  0.10000     0.21000     0.25000     0.41000     0.00000     0.35000     0.00000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.04600     0.05000     0.00000     0.10000     0.00000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.04600     0.05000     0.00000    -0.10000     0.00000     0.00000     0.00000     0.00000
  0.10000     0.04600     0.02300     0.05000    -0.08000    -0.60500     0.00000     0.00000     0.00000     0.00000
  0.10000     0.02300     0.02300     0.20000     0.00000    -0.60600     0.00000     0.00000     0.00000     0.00000
  0.10000     0.02300     0.04600     0.20000     0.06000    -0.60500     0.00000     0.00000     0.00000     0.00000


The logic of rotations by theta and psi angles is taken from the code of Matthias Schabel.
Other phantoms typically set theta=0 and psi=0 so there are no problems with different meanings of these angles in 
different authors.
"""

import numpy as np
from collections import namedtuple

Ellipsoid = namedtuple("Ellipsoid", "A a b c x0 y0 z0 phi theta psi")
Ellipse = namedtuple("Ellipse", "A a b x0 y0 phi")


def constructPhantom3D(e3d, n=512):
    p = np.zeros(n ** 3)
    rng = np.linspace(-1, 1, n)
    # The problem with the following is that when we later want to index
    # x[xind, yind, zind] so that x[xind,:,:] is constant as we apparently do
    # here then we have to use correct indexing moreover as in imagej axis y goes from top down, its better to 
    # flip it for better visualization
    x, y, z = np.meshgrid(rng, -rng, rng, indexing='ij')
    # Here we depend on internal alignment of numpy of flattened arrays and hope that flatten can be reversed by
    # reshape.  Note that for x[yind, xind, zind] indexing we would have to switch x and y and flipping x, y, z in 
    # meshgrid function would lead to more natural numpy alignment
    coord = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    p = p.flatten()

    for e in e3d:
        phi = np.radians(e.phi)  # first Euler angle in radians
        theta = np.radians(e.theta)  # second Euler angle in radians
        psi = np.radians(e.psi)  # third Euler angle in radians

        c1 = np.cos(phi)
        s1 = np.sin(phi)
        c2 = np.cos(theta)
        s2 = np.sin(theta)
        c3 = np.cos(psi)
        s3 = np.sin(psi)

        # Euler rotation matrix
        # Here Matthias Schabel evidently has choosen so called Z1X2Z3 matrix as defined in
        # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        # Implicit equation for ellipsoid is evaluated in rotated coordinates
        alpha = np.array(
            [[ c3 * c1 - c2 * s1 * s3, c3 * s1 + c2 * c1 * s3, s3 * s2],
             [-s3 * c1 - c2 * s1 * c3,-s3 * s1 + c2 * c1 * c3, c3 * s2],
             [ s2 * s1,               -s2 * c1,                c2]])

        # rotated ellipsoid coordinates
        coordp = np.matmul(alpha, coord)
        ellipsoidCenter = np.array([[e.x0], [e.y0], [e.z0]])
        ellipsoidSquareHalfaxes = np.array(np.square([[e.a], [e.b], [e.c]]))
        ellipsoidImplicitResult = np.sum(np.divide(np.square(np.subtract(         # Sum along columns
                                         coordp, ellipsoidCenter)), ellipsoidSquareHalfaxes), axis=0)
        p[ellipsoidImplicitResult <= 1.0] += e.A

    return np.transpose(np.clip(p, 0, 1).reshape(n, n, n), (2,1,0))  # [0,1] clip and index as [z,y,x]


def phantom3d(n=128, phantom='ToftSchabelKulvait3D'):
    """Three-dimensional Shepp-Logan like phantoms

    Can be used to test 3D reconstruction algorithms.

    Parameters
    ==========
        n : int, optional
            The grid size of the phantom
        phantom: str
                One of {'modified-shepp-logan', 'shepp_logan', 'yu_ye_wang'},
                The type of phantom to draw.

        Notes
        =====
        For any given voxel in the output image, the voxel's value is equal to the
        sum of the additive intensity values of all ellipsoids that the voxel is a
        part of.  If a voxel is not part of any ellipsoid, its value is 0.

        The additive intensity value A for an ellipsoid can be positive or
        negative;  if it is negative, the ellipsoid will be darker than the
        surrounding pixels.
        Note that, depending on the values of A, some voxels may have values
        outside the range [0,1].


    Copyright 2005 Matthias Christian Schabel (matthias @ stanfordalumni . org)
    University of Utah Department of Radiology
    Utah Center for Advanced Imaging Research
    729 Arapeen Drive
    Salt Lake City, UT 84108-1218

    This code is released under the Gnu Public License (GPL). For more information,
    see : http://www.gnu.org/copyleft/gpl.html

    Source modified by Vojtech Kulvait
    """
    if phantom == 'ToftSchabelKulvait3D':
        e3d = ToftSchabelKulvait3D()
    elif phantom == 'ToftSchabel3D':
        e3d = ToftSchabel3D()
    else:
        raise TypeError('phantom type "%s" not recognized' % phantom)
    return constructPhantom3D(e3d, n)


def EllipseToEllipsoid(ellipse):
    """Converts Ellipse to Ellipsoid where c=0, z0=0, theta=0, psi=0"""
    return Ellipsoid(ellipse.A, ellipse.a, ellipse.b, 0.0, ellipse.x0, ellipse.y0, 0.0, ellipse.phi, 0.0, 0.0)


def SheppLogan2D():
    """
    Shepp, L. A. & Logan, B. F.
    The Fourier reconstruction of a head section
    IEEE Transactions on Nuclear Science, Institute of Electrical and Electronics Engineers (IEEE), 1974, 21, 21-43
    """
    e2d = []
    e2d.append(Ellipse(2.0, 0.69, 0.92, 0.0, 0.0, 0.0))
    e2d.append(Ellipse(-0.98, 0.6624, 0.874, 0.0, -0.0184, 0.0))
    e2d.append(Ellipse(-0.02, 0.11, 0.31, 0.22, 0.0, -18))
    e2d.append(Ellipse(-0.02, 0.16, 0.41, -0.22, 0.0, 18))
    e2d.append(Ellipse(0.01, 0.21, 0.25, 0.0, 0.35, 0.0))
    e2d.append(Ellipse(0.01, 0.046, 0.046, 0.0, 0.1, 0.0))
    e2d.append(Ellipse(0.01, 0.046, 0.046, 0.0, -0.1, 0.0))
    e2d.append(Ellipse(0.01, 0.046, 0.023, -0.08, -0.605, 0.0))
    e2d.append(Ellipse(0.01, 0.023, 0.023, 0.0, -0.605, 0.0))
    e2d.append(Ellipse(0.01, 0.023, 0.046, 0.06, -0.605, 0.0))
    return e2d


def Toft2D():
    """Modified Shepp Logan from Octave"""
    e2d = SheppLogan2D()
    e2d[0] = e2d[0]._replace(A=1.0)
    e2d[1] = e2d[1]._replace(A=-0.8)
    e2d[2] = e2d[2]._replace(A=-0.2)
    e2d[3] = e2d[3]._replace(A=-0.2)
    e2d[4] = e2d[4]._replace(A=0.1)
    e2d[5] = e2d[5]._replace(A=0.1)
    e2d[6] = e2d[6]._replace(A=0.1)
    e2d[7] = e2d[7]._replace(A=0.1)
    e2d[8] = e2d[8]._replace(A=0.1, y0=-0.606)
    e2d[9] = e2d[9]._replace(A=0.1)
    return e2d


def ToftSchabel3D():
    """
    Specification of 10 ellipsoids given in Matthias Schabel (2020). 
    3D Shepp-Logan phantom contribution from modified_shepp_logan function
    Matthias Schabel (2020). 3D Shepp-Logan phantom
    https://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom
    """
    e2d = Toft2D()
    e3d = [EllipseToEllipsoid(x) for x in e2d]
    e3d[0] = e3d[0]._replace(c=0.81)
    e3d[1] = e3d[1]._replace(c=0.78)
    e3d[2] = e3d[2]._replace(c=0.22, psi=10.0)
    e3d[3] = e3d[3]._replace(c=0.28, psi=10.0)
    e3d[4] = e3d[4]._replace(c=0.41, z0=-0.15)
    e3d[5] = e3d[5]._replace(c=0.05, z0=0.25)
    e3d[6] = e3d[6]._replace(c=0.05, z0=0.25)
    e3d[7] = e3d[7]._replace(c=0.05)
    e3d[8] = e3d[8]._replace(c=0.2)
    e3d[9] = e3d[9]._replace(c=0.2)
    return e3d


def ToftSchabelKulvait3D():
    """Same as ToftShabel3D but z0 and psi are all zero """
    e3d = ToftSchabel3D()
    e3d[2] = e3d[2]._replace(psi=0.0)
    e3d[3] = e3d[3]._replace(psi=0.0)
    e3d[4] = e3d[4]._replace(z0=0.0)
    e3d[5] = e3d[5]._replace(z0=0.0)
    e3d[6] = e3d[6]._replace(z0=0.0)
    return e3d
