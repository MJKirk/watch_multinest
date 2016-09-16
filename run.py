"""
Run
===

Command-line interface for watching or snapshotting status of a MultiNest scan.
"""

from argparse import ArgumentParser as arg_parser
from snapshot import print_snapshot
from watch import watch

################################################################################


def __main__():
    """
    Monitor MultiNest scan via command line arguments.
    """

    # Make parser for command line arguments

    parser = arg_parser(description="Monitor MultiNest scan")

    parser.add_argument("root",
                        help="Prefix of MultiNest output filenames (root)",
                        type=str)
    parser.add_argument("--tol",
                        dest="tol",
                        help="MultiNest evidence tolerance factor (tol)",
                        type=float,
                        default=0.1,
                        required=False)
    parser.add_argument("--maxiter",
                        dest="maxiter",
                        help="MultiNest maximum number of iterations (maxiter)",
                        type=int,
                        default=float("inf"),
                        required=False)
    parser.add_argument("--watch",
                        dest="watch_mode",
                        help="Whether to watch rather than snapshot",
                        action='store_true',
                        default=False,
                        required=False)

    # Fetch arguments

    tol = parser.parse_args().tol
    assert tol > 0., "tol <= 0: %s" % tol
    maxiter = parser.parse_args().maxiter
    assert maxiter > 0, "maxiter <= 0: %s" % maxiter
    root = parser.parse_args().root
    watch_mode = parser.parse_args().watch_mode

    if watch_mode:
        watch(root, tol, maxiter)
    else:
        print_snapshot(root, tol, maxiter)


if __name__ == "__main__":
    __main__()
