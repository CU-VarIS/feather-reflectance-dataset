import numpy as np
from matplotlib import pyplot

from .dome_coordinates import DomeCoordinates
from .light_index import DomeLightIndex


def test_domeThetaPhi_to_envMapUV(di: DomeLightIndex, plot=False):
    uv = DomeCoordinates.t_domeThetaPhi_to_envMapUV(di.lights_domeThetaPhi)
    uv_orig = di.lights_envMapUV

    if plot:
        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 6))
        ax1.scatter(uv[:, 0], uv[:, 1], s=5, c=di.lights_dmx, label=di.lights_dmx)
        ax1.set_xlabel("u")
        ax1.set_ylabel("v")
        ax1.set_title("Reconstructed UV Coords")

        # plot original
        ax2.scatter(uv_orig[:, 0], uv_orig[:, 1], s=5)
        ax2.set_xlabel("u")
        ax2.set_ylabel("v")
        ax2.set_title("Original UV Coords")
        # pyplot.legend(ax1.get_legend_handles_labels()[0], di.lights_dmx, title='DMX')
        pyplot.legend(*ax1.get_legend_handles_labels())

        pyplot.show()
        pyplot.close(fig)

    np.testing.assert_allclose(uv, uv_orig, atol=1e-5)


def test_domeYUp_to_sample(di: DomeLightIndex, plot=False):
    domeYUp = di.lights_domeYUp
    sampleW0 = DomeCoordinates.t_domeYUp_to_sampleW0(domeYUp)

    def plot_axes(coords, dims: str, name):
        name_to_dim = {"X": 0, "Y": 1, "Z": 2}
        d0, d1 = name_to_dim[dims[0]], name_to_dim[dims[1]]

        # plot yz

        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 6))
        scatter = ax1.scatter(
            coords[:, d0], coords[:, d1], c=di.lights_dmx, cmap="tab10"
        )
        ax1.set_xlabel(dims[0])
        ax1.set_ylabel(dims[1])
        ax1.set_title(f"{name}: {dims}")
        # plot original
        scatter = ax2.scatter(
            coords[:, d0], coords[:, d1], c=di.lights_dmx, cmap="tab10"
        )
        ax2.set_xlabel(dims[0])
        ax2.set_ylabel(dims[1])
        ax2.set_title(f"{name}: {dims}")
        # top right legend
        pyplot.legend(*scatter.legend_elements(), title="DMX", loc="upper right")

        # pyplot.legend(scatter.legend_elements()[0], di.lights_dmx, title='DMX', loc='upper right')
        pyplot.show()
        pyplot.close(fig)

    if plot:
        plot_axes(domeYUp, "XY", "Dome Y Up")
        plot_axes(domeYUp, "ZY", "Dome Y Up")

        plot_axes(sampleW0, "XY", "Sample W0")
        plot_axes(sampleW0, "ZY", "Sample W0")

    # np.testing.assert_allclossampleW0e(domeYUp, di.lights_domeYUp, atol=1e-5)


def test_domeThetaPhi_to_domeYUp(di: DomeLightIndex, plot=False):
    domeYUp = DomeCoordinates.t_domeThetaPhi_to_domeYUp(di.lights_domeThetaPhi)

    if plot:
        # plot yz
        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 6))
        ax1.scatter(domeYUp[:, 1], domeYUp[:, 2], s=5)
        ax1.set_xlabel("Y")
        ax1.set_ylabel("Z")
        ax1.set_title("Reconstructed YZ Coords")

        # plot original
        ax2.scatter(di.lights_domeYUp[:, 1], di.lights_domeYUp[:, 2], s=5)
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")
        ax2.set_title("Original YZ Coords")
        pyplot.show()
        pyplot.close(fig)

    np.testing.assert_allclose(domeYUp, di.lights_domeYUp, atol=1e-5)


def test_envMapYUp_to_domeThetaPhi(di: DomeLightIndex, plot=False):
    domeThetaPhi = DomeCoordinates.t_envMapYUp_to_domeThetaPhi(di.lights_envMapYUp)

    if plot:
        # pyplot.rcParams.update({
        # 	"text.usetex": True,
        # })
        # plot yz
        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 6))
        ax1.scatter(domeThetaPhi[:, 0], domeThetaPhi[:, 1], s=5)
        ax1.set_xlabel("\\theta")
        ax1.set_ylabel("\\phi")
        ax1.set_title("Reconstructed \\theta\\phi Coords")

        # plot original
        ax2.scatter(di.lights_domeThetaPhi[:, 0], di.lights_domeThetaPhi[:, 1], s=5)
        ax2.set_xlabel("\\theta")
        ax2.set_ylabel("\\phi")
        ax2.set_title("Original \\theta\\phi Coords")
        pyplot.show()
        pyplot.close(fig)

    np.testing.assert_allclose(domeThetaPhi, di.lights_domeThetaPhi, atol=1e-5)


def test(plot=False):
    di = DomeLightIndex()
    test_domeThetaPhi_to_envMapUV(di, plot=plot)
    test_domeThetaPhi_to_domeYUp(di, plot=plot)
    test_envMapYUp_to_domeThetaPhi(di, plot=plot)
