"""
Multinest stopping criteria
===========================

Check whether a MultiNest scan is close to stopping by reading output
files.
"""

from __future__ import print_function
from numpy import log, exp, loadtxt, mean
from os.path import isfile
from warnings import warn
from pprint import pformat

PREAMBLE = """Four stopping criteria are applied per mode:

    mode_stop = 1. OR 2. OR 3. OR 4.

    1a. delta < tol
    1b. n_rejected - n_live > 50
    1. 1a. AND 1b.
    2. n_live_mode < n_dims + 1
    3. ln max_like - ln min_like <= 1E-3
    4. n_rejected >= max_iter

where we define delta = max_like * volume / evidence in a mode.

Once all modes have stopped, MultiNest stops.

Most modes eventually stop via criteria 1. 1b. is usually satisfied long
before 1a.

Monitor progression of scan by tracking progress of ln delta towards ln tol
per mode.
"""

################################################################################


def _error_ln_evidence(mode_):
    """
    :param mode: Information about a single mode
    :type mode: dict

    :returns: Error on log evidence
    :rtype: float
    """
    info = mode_["ln_Z_info"]
    n_live = mode_["n_live"]
    return (abs(info) / n_live)**0.5


_BOOL_STRING = lambda string: string == "T"

################################################################################


def print_snapshot(root, tol=float("inf"), maxiter=float("inf")):
    """
    Make and print snapshot of MultiNest scan.

    :param root: Prefix of MultiNest output filenames (root)
    :type root: string
    :param tol: MultiNest evidence tolerance factor (tol)
    :type tol: float
    :param maxiter: MultiNest maximum number of iterations (maxiter)
    :type maxiter: int
    """

    # Make snapshot

    information = snapshot(root, tol=tol, maxiter=maxiter)

    # Print information

    tformat = lambda title: (" " + title + " ").center(80, '=')

    print(tformat("Check MultiNest stopping criteria"), end="\n\n")
    print(PREAMBLE, end="\n\n")

    print(tformat("Global information"), end="\n\n")
    print(pformat(information["global"]), end="\n\n")

    # Print per mode information

    for mode_number, mode in sorted(information["modes"].items()):
        mode_info = "Mode: %s" % mode_number
        print(tformat(mode_info), end="\n\n")
        print(pformat(mode), end="\n\n")


def snapshot(root, tol=float("inf"), maxiter=float("inf")):
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
    assert tol > 0., "tol <= 0: %s" % tol
    assert maxiter > 0, "maxiter <= 0: %s" % maxiter

    # Dictionary for global information about results
    global_ = dict()

    # Fetch arguments
    global_["tol"] = tol
    global_["ln_tol"] = log(global_["tol"])
    global_["maxiter"] = maxiter
    global_["root"] = root

    # Check *resume.dat, *phys_live.points and *live.points
    resume_name = global_["root"] + "resume.dat"
    phys_live_name = global_["root"] + "phys_live.points"
    live_name = global_["root"] + "live.points"
    assert isfile(resume_name), "Cannot find: %s" % resume_name
    assert isfile(phys_live_name), "Cannot find: %s" % phys_live_name
    assert isfile(live_name), "Cannot find: %s" % live_name

    # Read data from *resume.dat, *phys_live.points and *live.points
    phys_live = loadtxt(phys_live_name, unpack=True)
    live = loadtxt(live_name, unpack=True)
    resume = map(str.split, open(resume_name))
    global_resume = resume[:4]  # General information
    modes_resume = resume[4:]  # Mode-specific information

    # Check first 4 lines of *resume.dat
    expected_shape = [1, 4, 2, 1]
    shape = map(len, global_resume)
    assert expected_shape == shape, "Wrong format: %s" % resume_name

    ############################################################################

    # Read information from *phys_live.points and *live.points

    global_["ln_max_like"] = max(phys_live[-2])
    global_["max_like"] = exp(global_["ln_max_like"])
    global_["expected_like"] = mean(exp(phys_live[-2]))
    global_["min_chi_squared"] = -2. * global_["ln_max_like"]
    global_["n_params"] = phys_live.shape[0] - 2
    global_["n_dims"] = live.shape[0] - 1

    ############################################################################

    # Read information about *global* evidence etc from *resume.dat

    # Read whether live-points generated
    global_["gen_live"] = _BOOL_STRING(global_resume[0][0])

    # Read number of rejected points
    global_["n_rejected"] = int(global_resume[1][0])
    assert global_["n_rejected"] >= 0

    # Read number of likelihood calls
    global_["n_like_calls"] = int(global_resume[1][1])
    assert global_["n_like_calls"] >= 0

    # Read number of modes
    global_["n_modes"] = int(global_resume[1][2])
    assert global_["n_modes"] >= 0

    # Read number of live points
    global_["n_live"] = int(global_resume[1][3])
    assert global_["n_live"] >= 0

    # Read total log evidence
    global_["ln_Z"] = float(global_resume[2][0])

    # Read error of total log evidence
    global_["ln_Z_info"] = float(global_resume[2][1])

    # Read whether ellipsoidal sampling
    global_["ellipsoidal"] = _BOOL_STRING(global_resume[3][0])

    ############################################################################

    # Read information about *modes* from *resume.dat

    modes = {m + 1: dict([["mode", m + 1]]) for m in range(global_["n_modes"])}

    for mode in modes.values():

        branch_line = modes_resume.pop(0)
        assert len(branch_line) == 1

        # Read number of branches in mode
        mode["branch_number"] = int(branch_line[0])
        assert mode["branch_number"] >= 0

        # Read unknown information about branches
        if mode["branch_number"]:

            branch_line = modes_resume.pop(0)
            assert len(branch_line) == 2

            mode["branch_unknown_1"] = str(branch_line[0])
            mode["branch_unknown_2"] = str(branch_line[1])
        else:
            mode["branch_unkown_1"] = mode["branch_unkown_2"] = None

    for mode in modes.values():

        mode_line = modes_resume.pop(0)
        assert len(mode_line) == 4

        # Read whether mode stopped
        mode["stop"] = _BOOL_STRING(mode_line[0])

        # Read unknown information about mode
        mode["mode_unknown_1"] = str(mode_line[1])
        mode["mode_unknown_2"] = str(mode_line[2])

        # Read number of live points in mode
        mode["n_live"] = int(mode_line[3])
        assert mode["n_live"] > 0

        mode_line = modes_resume.pop(0)
        assert len(mode_line) == 3

        # Read volume in mode
        mode["vol"] = float(mode_line[0])
        assert mode["vol"] >= 0.

        # Read log evidence in mode
        mode["ln_Z"] = float(mode_line[1])

        # Read error of log evidence in mode
        mode["ln_Z_info"] = float(mode_line[2])

        # Guess whether in constant efficiency mode by whether next line
        # is length 1
        if not "ceff" in global_:
            global_["ceff"] = bool(modes_resume) and len(modes_resume[0]) == 1

        # Read unknown information about constant efficiency mode
        if global_["ceff"]:

            mode_line = modes_resume.pop(0)
            assert len(mode_line) == 1

            mode["ceff_unknown"] = str(mode_line[0])

    # Should have parsed all lines
    assert not modes_resume, "Data not parsed: %s" % modes_resume

    ############################################################################

    # Extra global calculations

    global_["Z"] = exp(global_["ln_Z"])
    global_["ln_Z_error"] = _error_ln_evidence(global_)
    global_["Z_error"] = global_["ln_Z_error"] * global_["Z"]
    global_["stop_1b"] = global_["n_rejected"] - global_["n_live"] > 50
    global_["stop_4"] = global_["n_rejected"] >= global_["maxiter"]
    global_["stop"] = all([mode["stop"] for mode in modes.values()])
    if global_["stop"] and not global_["stop_1b"]:
        warn("Unusual convergence - very few rejected points")

    ############################################################################

    # Extra calculations per mode

    for n_mode, mode in modes.iteritems():

        # Column of log likelihood for this mode
        mode_ln_like = phys_live[:, phys_live[-1] == n_mode][-2]

        mode["ln_max_like"] = max(mode_ln_like)
        mode["max_like"] = exp(mode["ln_max_like"])
        mode["expected_like"] = mean(exp(mode_ln_like))
        mode["ln_min_like"] = min(mode_ln_like)
        mode["min_chi_squared"] = -2. * mode["ln_max_like"]

        mode["Z"] = exp(mode["ln_Z"])
        mode["ln_Z_error"] = _error_ln_evidence(mode)
        mode["Z_error"] = mode["ln_Z_error"] * mode["Z"]

        mode["ln_delta"] = log(mode["vol"]) + mode["ln_max_like"] - mode["ln_Z"]
        mode["delta"] = exp(mode["ln_delta"])
        mode["ln_expected_delta"] = log(mode["vol"]) + log(mode["expected_like"]) - mode["ln_Z"]
        mode["expected_delta"] = exp(mode["ln_expected_delta"])

        mode["stop_1a"] = mode["delta"] < global_["tol"]
        mode["stop_1b"] = global_["stop_1b"]
        mode["stop_1"] = mode["stop_1a"] and mode["stop_1b"]
        mode["stop_2"] = mode["n_live"] < global_["n_dims"] + 1
        mode["stop_3"] = mode["ln_max_like"] - mode["ln_min_like"] <= 1E-3
        mode["stop_4"] = global_["stop_4"]

        stop = (mode["stop_1"] or mode["stop_2"] or
                mode["stop_3"] or mode["stop_4"])
        assert mode["stop"] == stop, "Inconsistent convergence criteria!"

    ############################################################################

    # Combine information into a single dictionary
    information = {"modes": modes, "global": global_}

    return information
