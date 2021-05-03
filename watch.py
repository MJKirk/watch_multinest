"""
Watch
=====

Watch status of a MultiNest scan.
"""


from os.path import dirname, isdir, basename
from snapshot import snapshot, print_snapshot
from datetime import datetime, timedelta
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import inotify.adapters as inotify


def watch(root, tol=float("inf"), maxiter=float("inf")):
    """
    :param root: Prefix of MultiNest output filenames (root)
    :type root: string
    :param tol: MultiNest evidence tolerance factor (tol)
    :type tol: float
    :param maxiter: MultiNest maximum number of iterations (maxiter)
    :type maxiter: int

    :returns: All information about MultiNest scan
    :rtype: dict
    """

    # Find folder containing MultiNest output files
    folder = dirname(root)
    assert isdir(folder), "Cannot find: %s" % folder
    live_name = basename(root + "live.points")

    # Watch folder containing MultiNest output files
    watch_ = inotify.Inotify(block_duration_s=10)
    watch_.add_watch(folder)

    # Data-holders
    time_data = []
    ln_delta_data = []

    # Control of when to start fitting
    WAITING = True
    unique_ln_delta_values = 0

    time_start = datetime.now()
    print("Start time: %s" % time_start)

    for event in watch_.event_gen():

        if event and event[3] == live_name:

            # Snap MultiNest scan
            try:
                snap_time = datetime.now()
                snap = snapshot(root, tol, maxiter)
            except Exception as error:
                warn(str(error))
                continue

            # Record data about delta
            time_data.append(snap_time)
            ln_delta = [mode["ln_delta_max"] for mode in list(snap["modes"].values())]
            ln_delta_data.append(ln_delta)
            if WAITING:
                try:
                    if ln_delta_data[-2] != ln_delta_data[-1]:
                        unique_ln_delta_values += 1
                        print("Got", unique_ln_delta_values, "values, will start fitting at 10")
                except IndexError:
                    pass
                if unique_ln_delta_values >= 10:
                    WAITING = False

        else:
            
            if WAITING:
                continue
            
            # Make plot of progress
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            # ax.set_yscale('symlog')
            plt.gcf().autofmt_xdate()
            plt.xlabel(r'Time')
            plt.ylabel(r'$\ln \Delta \mathcal{Z}$')

            # Extrapolate time-remaining and plot, if possible
            time_float = [(t - time_start).total_seconds() for t in time_data]

            for n_mode, ln_delta in enumerate(map(list, list(zip(*ln_delta_data)))):

                fit = np.polyfit(ln_delta, time_float, 10)
                time_func = np.poly1d(fit)
                time_seconds = time_func(np.log(tol))
                guess_time_end = time_start + timedelta(seconds=time_seconds)

                estimate_seconds = [time_func(x) for x in ln_delta]
                estimate_time = [time_start + timedelta(seconds=e) for e in estimate_seconds]

                plt.plot(estimate_time + [guess_time_end], ln_delta + [np.log(tol)])
                plt.plot(time_data, ln_delta, "*")
                plt.axhline(np.log(tol), color="r")

                print("Mode: %s. Estimated end time: %s" %(n_mode, guess_time_end))

            plt.savefig("progress.png")
            plt.close()

            # Quit watching if MultiNest scan has stopped
            try:
                if snap["global"]["stop"]:
                    print_snapshot(root, tol, maxiter)
                    time_end = datetime.now()
                    delta_time = time_end - time_start
                    print("End time: %s" % time_end)
                    print("Total time: %s" % delta_time)
                    break
            except NameError:
                continue
