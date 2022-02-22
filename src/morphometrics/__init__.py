try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._sample_data import make_simple_labeled_cube, random_3d_image
