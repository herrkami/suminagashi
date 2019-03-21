'''
Copyright 2019 Korbinian Schreiber

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import seaborn as sbn

def rotate(vec, phi):
    '''
        Simple 2D  rotation of a vector vec around the origin by an angle phi.

        Parameters
        ----------
        vec : ndarray
            Vector(s) to rotate.
        phi : float
            Angle in radiants.

        Return
        ------
        out : ndarray
            Rotated vector(s).
    '''
    c, s = np.cos(phi), np.sin(phi)
    R = np.array(((c,-s), (s, c)))
    return np.dot(vec, R)

def softclip_pos(x, delta_x=1., x_max=0.):
    '''
        Apply quadratic soft clipping to the elements of x to a maximal value.

        Parameters
        ----------
        x : ndarray
            Array to be soft-clipped.
        delta_x : float
            Range over which to apply the soft-clipping.
        x_max : float
            Desired maximal value of the output data. +inf would be clipped to
            x_max.

        Return
        ------
        y : ndarray
            Soft-clipped array.
    '''
    f = lambda x: -(x - x_max - delta_x/2)**2/delta_x/2 + x_max
    y = np.zeros(x.shape)
    y = np.where(x - x_max < -delta_x/2, x, y)
    y = np.where(x - x_max > delta_x/2, x_max, y)
    y = np.where(np.logical_and(x - x_max >= -delta_x/2, x - x_max <= delta_x/2), f(x), y)
    return y

def softclip_neg(x, delta_x=1., x_min=-1.):
    '''
        Apply quadratic soft clipping to the elements of x to a minimal value.

        Parameters
        ----------
        x : ndarray
            Array to be soft-clipped.
        delta_x : float
            Range over which to apply the soft-clipping.
        x_min : float
            Desired minimal value of the output data. -inf would be clipped to
            x_min.

        Return
        ------
        y : ndarray
            Soft-clipped array.
    '''
    f = lambda x: (x - x_min + delta_x/2)**2/delta_x/2 + x_min
    y = np.zeros(x.shape)
    y = np.where(x > x_min + delta_x/2, x, y)  # linear
    y = np.where(x < x_min - delta_x/2, x_min, y)  # clip
    y = np.where(np.logical_and(x >= x_min - delta_x/2, x <= x_min + delta_x/2), f(x), y)  # transition
    return y

def scalescape(alpha=-0.5, phase=0, size=(100, 100)):
    '''
    Generate a fractal noise pattern by constraining the 2D power spectrum
    of white 2D noise to follow a power law with scaling constant alpha.

    Parameters
    ----------
    alpha : float, optional
        Scaling constant. Should be negative.
    phase : float, optional, experimental
        Fourier phase of the data set. Cyclic within [0, 2*pi].
    size : tuple of two ints
        Size of the pattern.
    '''
    scape = np.random.uniform(0, 1, size=size)
    rfft = np.fft.rfft2(scape)
    my, mx = rfft.shape  # reverse coordinates for meshgrid
    mx, my = np.meshgrid(np.linspace(0, 1, mx), np.linspace(-1, 1, my))/np.sqrt(2)
    r = mx**2 + my**2
    f = r**alpha
    rfft = np.roll(rfft, rfft.shape[0]//2 - 1, axis=0)
    # Apply filter
    ffft = np.multiply(rfft, f)
    # Phase shift
    ffft *= np.exp(1.j*phase)
    ffft = np.roll(ffft, 1 - rfft.shape[0]//2, axis=0)
    # Invert ffft
    scape = np.fft.irfft2(ffft)
    # Normalize to [0, 1]
    scape = ((scape - np.min(scape))/(np.max(scape) - np.min(scape)))
    return scape

def suminagashi(nr_rings=21,
                r_max=0.4,
                nr_sources=3,
                nr_phis=1000,
                iterations=20,
                resolution=20,
                alpha=-0.9,
                wall_repulsion=0.5,
                colorscheme=['black', 'white'],
                filename=None,
                plot=False,
                series=False,
                seed=None,
                dpi=300):
    '''
        Create a suminagashi figure.

        Parameters
        ----------
        nr_rings : int, optional, default is 21
            Number of rings per source.

        r_max : float, optional, default is 0.4
            Maximal radius of the rings around one source. Should be between 0
            and 1.

        nr_sources : int, optional, default is 3
            Number of sources.

        nr_phis : int, default is 1000
            Number of angles per ring.

        iterations : int, optional, default is 20
            Number of warping iterations. Initialy, the rings are concentric
            around the source but compressed to not touch the area edges or
            each other.

        resolution : int, optional, default is 20
            Resolution of the vector field that does the warping. Since linear
            interpolation is used to calculate the vector field at an arbitray
            position, the resolution is not as crucial as it might seem and 20
            is quite enough.

        alpha : float, optional, default is -0.9
            Power law scaling factor of the potential that the vector field is
            derived from. See scapescale() help for reference. Should be
            negative around -1.0.

        wall_repulsion : float, optional, default is 0.5
            Range of the wall repulsion relative to the total size of the
            vector field. 1 would cover the whole field.

        colorscheme : list of color strings or seaborn colorschemes, optional,
                      default is ['black', 'white']
            You can parse any list of colors or a colorscheme of your choice.
            E.g., ['#1A191A', '#253942', '#E8F0F4', '#FFFFFF', '#F05216'], or
            sbn.set_palette(sbn.cubehelix_palette(n_colors=2), 3).

        filename : string, optional, default is None
            Filename of the figure. If None is given, the picture is not saved.

        plot : bool, optional, default False
            If True, the plot will be shown with plt.show().

        series : bool, optional, default False
            If True, a full series of pictures of the whole warping process
            will be plotted.

        seed : int or 1-d array_like, optional
            Seed for numpy `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        dpi : int, optional, default is 300
            Picture resolution.

        Return
        ------
        y : ndarray
            Soft-clipped array.
    '''

    # Set the random seed
    np.random.seed(seed)

    # Set the color scheme
    sbn.set_palette(colorscheme)

    # Create the morphing potential
    scape = scalescape(alpha=alpha, size=(resolution, resolution))

    # Get vector field (approaching minima)
    x = np.arange(resolution)
    y = np.arange(resolution)
    gradx, grady = np.gradient(scape)
    gradx_interp = RegularGridInterpolator((x, y), -gradx)
    grady_interp = RegularGridInterpolator((x, y), -grady)

    # Create unmorphed rings
    sources = np.random.randint(0.2*resolution, 0.8*resolution, (nr_sources, 2))
    rings = np.zeros((nr_sources, nr_rings, nr_phis, 2))
    for s, source in enumerate(sources):
        rad = r_max
        radii = np.sort(np.random.uniform(rad*resolution,
                                          0.01*rad*resolution,
                                          nr_rings))
        radii = radii[::-1]
        phis = np.linspace(0, 2*np.pi, nr_phis, endpoint=True)
        for i, r in enumerate(radii):
            rings[s, i, :, :] = np.array([r*np.cos(phis) + source[0], r*np.sin(phis) + source[1]]).T

    # Separate the rings by soft clipping
    for s, source in enumerate(sources):
        for other in [o for j, o in enumerate(sources) if j!=s]:
            vec = np.array(other) - source
            length = np.sqrt(vec[0]**2 + vec[1]**2)
            phi = np.angle(vec[0] + vec[1]*1.j)
            for r, ring in enumerate(rings[s, :, :, :]):
                ring = rotate(ring, phi)
                x_max = rotate(source, phi)[0] + length/2
                ring[:, 0] = softclip_pos(ring[:, 0], delta_x=r_max*0.5*resolution, x_max=x_max)
                ring = rotate(ring, -phi)
                rings[s, r, :, :] = ring

    # Separate the rings from the walls
    for s, source in enumerate(sources):
        for r, ring in enumerate(rings[s, :, :, :]):
            ring[:, 0] = softclip_pos(ring[:, 0], delta_x=0.5*resolution, x_max=resolution-1)
            ring[:, 0] = softclip_neg(ring[:, 0], delta_x=0.5*resolution, x_min=0)
            ring[:, 1] = softclip_pos(ring[:, 1], delta_x=0.5*resolution, x_max=resolution-1)
            ring[:, 1] = softclip_neg(ring[:, 1], delta_x=0.5*resolution, x_min=0)


    # Warp the rings
    for i in range(iterations):
        for s in range(nr_sources):
            for r, ring in enumerate(rings[s, :, :, :]):
                # Morph them
                rings[s, r, :, 0] += gradx_interp(np.mod(ring, resolution-1))
                rings[s, r, :, 1] += grady_interp(np.mod(ring, resolution-1))
                # Apply wall constraint
                rings[s, r, :, 0] = softclip_pos(ring[:, 0], delta_x=wall_repulsion*resolution, x_max=resolution-1)
                rings[s, r, :, 0] = softclip_neg(ring[:, 0], delta_x=wall_repulsion*resolution, x_min=0)
                rings[s, r, :, 1] = softclip_pos(ring[:, 1], delta_x=wall_repulsion*resolution, x_max=resolution-1)
                rings[s, r, :, 1] = softclip_neg(ring[:, 1], delta_x=wall_repulsion*resolution, x_min=0)
        if series:
            fig = plt.figure(frameon=False, figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            for s in range(nr_sources):
                for r, ring in enumerate(rings[s, :, :, :]):
                    ax.fill(ring[:, 0], ring[:, 1], alpha=1)
            if filename is not None:
                fig.savefig('{}_series{:04d}.png'.format(filename, i),
                            dpi=dpi, pad_inches=0)
            plt.close()

    # Create the figure
    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    for s in range(nr_sources):
        for r, ring in enumerate(rings[s, :, :, :]):
            ax.fill(ring[:, 0], ring[:, 1], alpha=1)

    if filename is not None:
        fig.savefig(filename, dpi=dpi, pad_inches=0)

    if plot:
        plt.show()

    plt.close()

if __name__ == '__main__':
    suminagashi(nr_rings=81,
                r_max=0.4,
                nr_sources=3,
                nr_phis=1000,
                iterations=20,
                resolution=20,
                alpha=-1.0,
                wall_repulsion=0.5,
                colorscheme=['black', 'white'],
                # colorscheme=sbn.set_palette(sbn.cubehelix_palette(n_colors=3, rot=0.9, hue=0.4), 3),
                filename='png/tmp.png',
                plot=True,
                series=False,
                seed=None,
                dpi=300)
