"""
@author: Adrien F. Vincent
@date: 2022-03-07
@version: 0.4.0

Description
===========
Helper script for interactively estimating the average values of data logged
by s-tui [1] (on Linux) or by Intel Power Gadget [2] (on OSX or Windows) or
by MX Power Gadget [3] (for ARM-based Apple computers) about the power
consumption, the processor utilization percentage as well as its frequency.

[1](https://amanusk.github.io/s-tui/)
[2](https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html)
[3](https://www.seense.com/menubarstats/mxpg/)

Usage
=====
One can use it directly through a command, e.g.
```
# S-tui (one needs to provide the sampling interval value)
python3 ./power_analysis_tools.py -s stui -f ./my_log.csv -dt 2.0
# Intel Power Gadget
python3 ./power_analysis_tools.py -s intelpowergadget -f ./my_log.csv
# MX Power Gadget
python3 ./power_analysis_tools.py -s mxpowergadget -f ./my_log.csv
```
or from within IPython:
```
# S-tui (one needs to provide the sampling interval value).
%run ./power_analysis_tools.py -s stui -f ./my_log.csv -dt 2.0
# Intel Power Gadget
%run ./power_analysis_tools.py -s intelpowergadget -f ./my_log.csv
# MX Power Gadget
%run ./power_analysis_tools.py -s mxpowergadget -f ./my_log.csv
```
The IPython command accepts the same arguments the bare CLI versions.

One could also want to modify some of the parameters at the top of the
script file depending on one's computer and the configuration of the
logging software configuration in order to avoid having to provide most
of the command arguments.

NB: to save the s-tui data into an appropriate log file, one can use the
following command (assuming that the current folder is the same a the one
where the current Python script is run):
```
sudo s-tui --csv-file ./my_log.csv
```
Sudo is required in order to have access to the power consumption information.

Besides, one can save the current s-tui configuration directly from s-tui,
which can for example be used to setup the refresh time interval at a lower
value than the default one (e.g. 0.25 s instead of 2 s) once and for all.
Beware however of not using a refresh time interval that is so short that
s-tui becomes an non negligible computing load.

Command arguments
=================
```
python3 ./power_analysis_tools.py --help
```

Requirements
============
Here are the oldest versions of the required packages that the script has
been successfully tested with:
* Python 3.6 (will not work with older versions due to the lack of f-strings)
* Numpy 1.19 (should be fine with much older versions)
* Matplotlib 3.3 (*might* be fine with the 3.2 version)
* Argparse 1.1 (should be fine with much older versions)

Miscellaneous
=============
After using the zoom tool (in the figure toolbar), one needs to click on the
zoom button again in order to make the selection of data points active again.

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FIGNUM = "average_estimate"

STUI_KEY = "stui"
INTEL_PG_KEY = "intelpowergadget"
MX_PG_KEY = "mxpowergadget"

POWER_KEY = "Power (W)"
UTILIZATION_KEY = "Utilization (%)"
FREQUENCY_KEY = "Frequency (MHz)"
TIME_KEY = "Time (s)"

# The following dictionnary defines the options of the CSV log files
# produced by the different logging softwares. In case the number of
# summary lines at the end of Intel Power Gadget logs were actually
# different in logs recorded on different computers, one can modify
# the `skip_footer` value accordingly to the relevant amount of lines.
DEFAULT_CSV_OPTIONS = {
    STUI_KEY: {  # s-tui
        "delimiter": ",",
        "skip_header": 1,
    },
    INTEL_PG_KEY: {  # Intel Power Gadget
        "delimiter": ",",
        "skip_header": 1,
        "skip_footer": 19,  # modify if more or less lines in your logs
                            # May actually be 15 lines on MacOS (Collin's log)
                            # vs. 19 lines on Windows (Jego's log)
    },
    MX_PG_KEY: {  # MX Power Gadget
        "delimiter": ",",
        "skip_header": 1,
    },
}
# The following dictionnaries define the names of the columns that
# one is going to look after in the log files. If those names were
# actually different in logs recorded on different computers, one can
# modify them either directly here in the source code or using the
# relevant command line arguments. See the output of the command
# `python3 ./power_analysis_tools.py -- help` for more information on
# the latter option.
#
# Regarding the case of ARM processors for Apple computers (thus for
# MX Power Gadget logs), one focuses on the P-core (Performance) rather
# than the E-core (Efficiency) as that is the one that is the more
# likely to be heavily used during the ENSEIRB-Matmeca labworks (about
# computation intensive-tasks).
#
# Names of the columns to target regarding y-axis data
DEFAULT_TARGETED_CSV_YDATA_COLUMN_NAMES = {
    STUI_KEY: {  # s-tui
        POWER_KEY: "Power:package-0,0",
        UTILIZATION_KEY: "Util:Avg",
        FREQUENCY_KEY: "Frequency:Avg",
    },
    INTEL_PG_KEY: {  # Intel Power Gadget
        POWER_KEY: "Processor Power_0(Watt)",
        UTILIZATION_KEY: "CPU Utilization(%)",
        FREQUENCY_KEY: "CPU Frequency_0(MHz)",
    },
    MX_PG_KEY: {  # MX Power Gadget
        POWER_KEY: "Package Power (Watt)",
        UTILIZATION_KEY: "P-Core Utilization (%)",
        FREQUENCY_KEY: "P-Core Frequency (Mhz)",
    },
}
# Name of the column to target regarding x-axis data
DEFAULT_TARGETED_CSV_XDATA_COLUMN_NAME = {
    STUI_KEY: None,  # actually not available in s-tui logs
    INTEL_PG_KEY: "Elapsed Time (sec)",  # for Intel Power Gadget logs
    MX_PG_KEY: "Elapsed Time (sec)",  # for MX Power Gadget logs
}



class AverageEstimator:
    """
    Class to interact with a plotted line and estimate its average value
    between two points defined by the user with the mouse.

    """

    def __init__(self, line, x_units="#"):
        """
        Instantiates an `AverageEstimator` object that can be used to
        interactively get an estimation of the average value between
        two points of a plotted line.

        Parameters
        ----------
        line : matplotlib.lines.Line2D
            The line to track regarding data picking events.  NB: the
            picker behavior of the line instance must have been initialized
            beforehand, e.g. `ax.plot(..., picker=True, pickradius=5)`.
        x_units : str, optional
            The units of the x-axis. The default is "#".

        Returns
        -------
        None.

        """
        self.x_units = x_units
        self.line = line
        self.cid = line.figure.canvas.mpl_connect("pick_event", self)
        self.picked_xs = [None, None]
        self.active_pick_index = 0
        self.x_delta = None
        self.y_average = None
        self._x_delta_str = ""
        self._y_avg_str = None

        _sty = {"lw": self.line.get_linewidth() + 0.5,
                #"color": self.line.get_color(),
                "color": "tab:red",
                "zorder": self.line.get_zorder() + 1,
                }
        self._selected_line, = self.line.axes.plot([], [], **_sty)
        self._selected_edges, = self.line.axes.plot([], [],
                                                    marker="d", ls="", **_sty)

    def __call__(self, event):
        """
        What to do when a relevant (line picking) event occurs.

        """
        thisline = event.artist
        event_occured_in_this_ax = event.mouseevent.inaxes is self.line.axes
        x_data = np.array(thisline.get_xdata())
        y_data = np.array(thisline.get_ydata())
        # The median value should provide a reasonable estimate of the
        # central x-axis of the picking point even if there is a data step:
        ind = int(np.median(event.ind))
        self.picked_xs[self.active_pick_index] = x_data[ind]

        if self.active_pick_index < 1:

            # Clear subplot title while a valid estimate is not available
            self.line.axes.set_title("")

        else:  # Time to perform a few computations!

            subplot_title = ""

            if not event_occured_in_this_ax:

                self._selected_line.set_data([], [])
                self._selected_edges.set_data([], [])

            else:

                # Swap the coordinates if x-values are not increasing
                if self.picked_xs[0] > self.picked_xs[1]:
                    self.picked_xs[:] = self.picked_xs[::-1]

                # Get x-index of the data points closest to the picked values
                x0_idx = np.argmin(np.abs(x_data - self.picked_xs[0]))
                x1_idx = np.argmin(np.abs(x_data - self.picked_xs[1]))
                # Let us be defensive and avoid null-length selections.
                # Anyway, without the increase, it looks like the selection
                # range stops one sample too early for some reason...
                x1_idx += 1

                print(f"=== {self.line.axes.get_ylabel()} ===")


                # Estimate the y-average.
                # NB: the current computation avoids using Scipy or a custom
                # reimplementation of numerical integration by relyong on the
                # fact that the measurement uses regular sampling.
                self.y_average = np.mean(y_data[x0_idx:x1_idx])
                self._y_avg_str = f"{self.y_average:0.3f} units"
                print(f"Average y-value: {self.y_average:0.3f} units.")

                # # Display the x-delta value
                self.x_delta = x_data[x1_idx] - x_data[x0_idx]
                _tmp = f"{self.x_delta:0.3f} {self.x_units}"
                _tmp = _tmp.rstrip()  # if no unit, remove the pending space
                print(f"Interval:\t {_tmp}.\n")
                self._x_delta_str = f" ({_tmp})"

                # Emphasize the selected data segment
                if event_occured_in_this_ax:
                    _x_vals = x_data[x0_idx:x1_idx]
                    _y_vals = y_data[x0_idx:x1_idx]
                    self._selected_line.set_data(_x_vals, _y_vals)
                    self._selected_edges.set_data((_x_vals[0], _x_vals[-1]),
                                                  (_y_vals[0], _y_vals[-1]))

                # Update the subplot title
                if self.y_average is not None:
                    subplot_title = f"Average: {self._y_avg_str}{self._x_delta_str}."

            self.line.axes.set_title(subplot_title)

        # Keep track of where one is regarding picking events
        #self.active_pick_index = int(not self.active_pick_index)
        self.active_pick_index = 1 - self.active_pick_index  # more common?

        # Update the display
        self.line.figure.canvas.draw_idle()



class LogContainer(object):
    """
    Class to load, store and manipulate data that is relevant to power,
    utilization and frequency monitoring.

    """
    def __init__(self, software, time_col_name=None, power_col_name=None,
                 util_col_name=None, freq_col_name=None):
        """
        Instantiates a `LogContainer` object that can be used for handling
        monitoring data logs.

        Parameters
        ----------
        software : str in ('{STUI_KEY}', '{INTEL_PG_KEY}', '{MX_PG_KEY}')
            The monitoring software that recorded the target log file.
        time_col_name : str or None, optional
            The targeted name for the column related to elapsed time (in s).
            NB: s-tui log fils may not have such a column. The default is None.
        power_col_name : str or None, optional
            The targeted name for the column related to power consumption
            (in W). A string "none" means not to use such data. The default
            is None (beware, not "none").
        util_col_name : str or None, optional
            The targeted name for the column related to processor utilization
            (in %). A string "none" means not to use such data. The default
            is None (beware, not "none").
        freq_col_name : str or None, optional
            The targeted name for the column related to clock frequency
            (in MHz). A string "none" means not to use such data. The default
            is None (beware, not "none").

        Raises
        ------
        ValueError
            The *software* value is unknown.

        Remarks
        -------
        If some arguments are None, then the corresponding values from
        either `DEFAULT_TARGETED_CSV_XDATA_COLUMN_NAME` or
        `DEFAULT_TARGETED_CSV_YDATA_COLUMN_NAMES` are used.

        Returns
        -------
        None.

        """
        if software.lower() in (STUI_KEY, INTEL_PG_KEY, MX_PG_KEY):
            self.software = software.lower()
        else:
            raise ValueError(f"Unknown software {software}.")

        self.csv_options = DEFAULT_CSV_OPTIONS[self.software]

        self.filepath = None  # to keep track of the (latest) loaded file

        self.dt = None

        self.targeted_csv_xdata_column_name = DEFAULT_TARGETED_CSV_XDATA_COLUMN_NAME[self.software]
        if (time_col_name is not None) and (self.software != STUI_KEY):
            self.targeted_csv_xdata_column_name = time_col_name

        self.targeted_csv_ydata_column_names = {**DEFAULT_TARGETED_CSV_YDATA_COLUMN_NAMES[self.software]}
        if power_col_name is not None:
            _name = power_col_name if power_col_name.lower() != "none" else None
            self.targeted_csv_ydata_column_names[POWER_KEY] = _name
            if _name is None:
                print("NB: Skipping the search for power data.")
        if util_col_name is not None:
            _name = util_col_name if util_col_name.lower() != "none" else None
            self.targeted_csv_ydata_column_names[UTILIZATION_KEY] = _name
            if _name is None:
                print("NB: Skipping the search for utilization data.")
        if freq_col_name is not None:
            _name = freq_col_name if freq_col_name.lower() != "none" else None
            self.targeted_csv_ydata_column_names[FREQUENCY_KEY] = _name
            if _name is None:
                print("NB: Skipping the search for frequency data.")

        self.csv_xdata_column_name = None
        self.csv_ydata_column_names = None
        self.csv_xdata_column_index = None
        self.csv_ydata_column_indices = None

        self._xdata_label = None
        self._ydata_labels = None
        self.xdata = None
        self.ydata = None

        self.fig = None
        self.axs = None

    def _get_column_names(self, filepath):
        """
        Returns the names of the columns of a log file. This method is
        especially useful with s-tui log files, where the column names
        may include commas that are also the CSV delimiter... Without
        a bit of massaging, one could thus have trouble finding the
        indices of the columns to load later.

        Parameters
        ----------
        filepath : str or path
            The path toward the data log file to use.

        Returns
        -------
        all_column_names : tuple of str
            The names of all the columns of the log file, from left to right.

        """
        # Load only the first raw of the file as a sequence of string in order
        # to get the (raw) names of the columns.
        array_header = np.genfromtxt(filepath, max_rows=1, dtype=str,
                                     delimiter=self.csv_options.get("delimiter", None))

        # Get the real names of the columns. More or less only required with
        # s-tui log files as some column names are for example "col_name,0":
        # the coma was interpreted like a relevant delimiter (which it is not).
        # One thus need to put everything back together
        all_column_names = []
        previous_element = None
        for new_element in array_header:

            # Some extra quote characters (") or spaces seems to be present
            # sometimes: let us do some cleaning before doing anything else.
            new_element = new_element.strip(' "')  # *space is important here*

            if previous_element is None:  # -> first iteration
                previous_element = new_element
                continue

            try:
                # Try out if the new element under scrutiny may be an integer
                # that got separate from its complet column name when (all)
                # commas have been intrepreted as column delimiters. If it is,
                # concatenate it with the previous element, while not
                # forgetting to add the aforementioned comma.
                _looks_like_an_index = int(new_element)  # if not: raises
                previous_element = f"{previous_element},{new_element}"
                continue  # just in case of a hell scenario like "col_name,1,0"
            except ValueError:
                try:
                    # See if it looks like a string name end (based on
                    # tests with Y. Bornat's log file example).
                    if "pkg" in new_element.lower():
                        previous_element = f"{previous_element},{new_element}"
                        continue
                except:
                    pass

            # The previous element seems (now) to be complete and the
            # new element does not looks like a part of an index.
            all_column_names.append(previous_element)
            previous_element = new_element

        # Do not forget if the last column name
        if (previous_element) != 0:
            all_column_names.append(previous_element)

        return tuple(all_column_names)

    def load(self, filepath, dt=None):
        """
        Load a relevant data from a log file recorded by the `self.software`
        software. The names of the target columns to use in the log file
        should already have been defined when instanciating `LogContainer`
        and are:
            - `self.time_col_name` for elapsed time (in s);
            - `self.power_col_name` for power consumption (in W);
            - `self.util_col_name` for processor utilization (in %);
            - `self.freq_col_name` for clock frequency (in MHz).

        Parameters
        ----------
        filepath : str or path
            The path toward the data log file to use.
        dt : float or None, optional
            The (regular) time interval between samples (in s). If there
            is a proper column defined in the log file regarding the
            elapsed time, then the latter has the priority over any *dt*
            value. If it is None and there is no proper column defined in
            the log file regarding the elapsed time, then a regularly space
            time vector is created based on the *dt* value and the number
            of samples. The default is None.

        Raises
        ------
        ValueError
            Some of the columns targeted in the log could not be found.

        Returns
        -------
        None.

        """
        self.dt = dt
        if (self.dt is None) and (self.software == STUI_KEY):
            raise ValueError("*dt* value cannot be None with s-tui logs.")

        self.filepath = filepath
        print(f"Loading the {self.software} log file:\n{self.filepath}...")

        all_column_names = self._get_column_names(self.filepath)

        # Load y-data first so that one knows the size of the required
        # array for time if one needed it as with s-tui log files.


        # Look for the index of the time column (x-data) and load it
        # (Power Gadget log file) or reconstruct it (s-tui log file).
        if self.targeted_csv_xdata_column_name is not None:
            try:
                name = self.targeted_csv_xdata_column_name
                self.csv_xdata_column_index = all_column_names.index(name)
            except ValueError:
                if self.dt is None:
                    raise ValueError(f"Could not find a column named '{name}' in {all_column_names}.")
                # If a 'dt' value is provided, let us try to reconstruct a
                # plausible time vector based on that information
                print(f"Warning: could not find a column named '{name}' in {all_column_names}.")
                print(f"Trying to make use of the provided `dt` value: {self.dt} (in s).\n")
                self.csv_xdata_column_index = None
                self.csv_xdata_column_name = None
        else:
            self.csv_xdata_column_index = None

        # Load CSV file

        # Y-data column selection: similar in both softwares
        self.csv_ydata_column_indices = []  # reset
        self.csv_ydata_column_names = []  # reset
        self._ydata_labels = []  # reset
        for key in self.targeted_csv_ydata_column_names:
            name = self.targeted_csv_ydata_column_names[key]
            try:
                if name is not None:  # i.e. one wants to use such data
                    index = all_column_names.index(name)
                    self.csv_ydata_column_indices.append(index)
                    self.csv_ydata_column_names.append(name)  # name in log file
                    self._ydata_labels.append(key)  # normalized name
            except ValueError:
                print(f"Warning: could not find a column named '{name}' in {all_column_names}.\n")

        if len(self._ydata_labels) == 0:  # something went horribly wrong
            raise ValueError("Y-data: could not find any of the targeted columns in log file.")

        # X-data column selection: depends on the software
        self.csv_xdata_column_index = None  # reset
        self.csv_xdata_column_name = None  # reset
        self._xdata_label = TIME_KEY  # normalized name
        if self.software == STUI_KEY:

                self.csv_xdata_column_index = None
                self.csv_xdata_column_name = None

        elif self.software in (INTEL_PG_KEY, MX_PG_KEY):

            name = self.targeted_csv_xdata_column_name

            try:
                index = all_column_names.index(name)
                self.csv_xdata_column_index = index
                self.csv_xdata_column_name = name  # name in log file
            except ValueError:
                if self.dt is None:
                    raise ValueError(f"Could not find a column named '{name}' in {all_column_names}.")
                # If a 'dt' value is provided, let us try to reconstruct a
                # plausible time vector based on that information
                print(f"Warning: could not find a column named '{name}' in {all_column_names}.")
                print(f"Trying to make use of the provided `dt` value: {self.dt} (in s).\n")
                self.csv_xdata_column_index = None
                self.csv_xdata_column_name = None


        # Load the relevant columns.
        relevant_columns = tuple(self.csv_ydata_column_indices)
        if self.csv_xdata_column_index is not None:
            relevant_columns += (self.csv_xdata_column_index,)  # comma is important
        # NB: one relies on `numpy.genfromtxt` rather than the more usual
        # `np.loadtxt` because Power Gadget log files contains summary
        # lines at the end that do not have the same amount of columns as
        # the rest of the (actually) relevant data. Thus the presence of
        # the *skip_footer* entry in the CSV options for Power Gadget files.
        # NB2: on log files produced by MacOS, all columns seems to be
        # typed as string values (with '"' characters and extra spaces.
        # Thus the use of the `encoding=None` and `dtype=str` options
        # when calling `numpy.genfromtxt` as well as the crude massaging
        # of the data array after loading the raw values.
        arr = np.genfromtxt(self.filepath, usecols=relevant_columns,
                                encoding=None, dtype=str,
                                **self.csv_options)
        arr = np.array([float(v.strip('"'))  # typ. for files recorded by MacOS
                        for v in arr.flatten()]).reshape(arr.shape)

        if arr.ndim == 1:
            # One needs to reshape the 1d-vector to a 2d-but-actually-a-
            # single-column array to avoid problems afterwards regarding
            # the selection of the relevant y-data for plotting.
            arr = arr.reshape((-1, 1))

        # Assign the relevant vectors to the x- and y-data containers.
        # Y-data
        # WIP: would a list of arrays be better for storing the y-data (more
        # similar to what is done for the singla x-data vector and might
        # avoid the redundant information of the keys with respect to the
        # `self.ydata_labels`.
        self.ydata = {}  # reset
        for _i, _k in enumerate(self._ydata_labels):
            self.ydata[_k] = arr[:, _i]
        # X-data
        self.xdata = {}  # reset
        if self.csv_xdata_column_index is not None:
            self.xdata[self._xdata_label] = arr[:, -1]
            # WIP: should one use relative time?
        else:
            self.xdata[self._xdata_label] = np.arange(arr.shape[0]) * self.dt

        print("Done!")

    def plot(self, fignum=DEFAULT_FIGNUM):
        f"""
        Plot the last dataset that has been load by `self.load()`.

        Parameters
        ----------
        fignum : int or str or None, optional
            The figure ID number or label. The default is {DEFAULT_FIGNUM}.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axs : array of matplotlib.axes.Axes

        """
        # Convenience variables
        ydata = self.ydata
        xdata = self.xdata
        xkey = list(self.xdata.keys())[0]  # xdata dict has a single key
        xlabel = xkey

        # Setup a (brand new) figure
        plt.close(fignum)  # hard clear. Dirty way to disconnect any callback.
        _cmn_fig_opts = {"nrows": len(ydata),
                         "num": fignum,
                         "sharex": True,
                         "clear": True,
                         "squeeze": False,  # Or otherwise one might need
                         }                  # special casing for a single panel
        try:
            fig, axs = plt.subplots(**_cmn_fig_opts, layout="constrained")
        except TypeError:
            # Looks like layout="constrained" is pretty new so give it a shot with
            # the now deprecated way (at least in Matplotlib 3.5) just in case.
            fig, axs = plt.subplots(**_cmn_fig_opts, constrained_layout=True)
        fig.align_ylabels()
        fig.avg_estimator = {}  # user-defined attribute (for average estimators)

        # Plot one set of data (i.e. a curve) per entry in the data dictionary
        line_options = {"linewidth": 1.0,
                        "linestyle": "solid",
                        "color": "tab:blue",
                        # Options related to picking
                        "picker": True,
                        "pickradius": 5,  # in points
                        }
        x_vals = xdata[xkey]
        for (key, ax) in zip(ydata, axs.flat):

            # Plot a new line plot (with picking capability activated)
            y_vals = ydata[key]
            line, = ax.plot(x_vals, y_vals, **line_options)
            fig.avg_estimator[key] = AverageEstimator(line, x_units="s")
            # WIP: the dx parameters is currently needed inside the
            # AverageEstimator code but there is no reason that we
            # may not get rid of that need in our case here.

            # Cosmeticks
            ax.set_xlabel(xlabel)
            ax.set_xlim(x_vals[0], x_vals[-1])  # remove default extra white space
            ax.set_ylabel(key)
            ax.minorticks_on()
            ax.label_outer()  # keep only outer (here left and bottom) labels
            ax.grid(which="major", axis="both", color="0.65", lw=0.8, ls=":")
            ax.grid(which="minor", axis="both", color="0.85", lw=0.8, ls=":")

        return fig, axs

def parse_arguments():
    """
    Parse CLI arguments. Use `-h` or `--help` for more information.

    Returns
    -------
    args : argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--filepath", type=str, required=True,
        #default=DEFAULT_FILEPATH,
        help="Path where the CSV log file is located.")
    parser.add_argument(
        "-s", "--software", type=str, required=True,
        choices=[STUI_KEY, INTEL_PG_KEY, MX_PG_KEY],
        #default=DEFAULT_SOFTWARE,
        help="Software used for the data recording.")
    parser.add_argument(
        "-dt", "--delta_t", type=float,
        #default=DEFAULT_DX,
        help="(Regular) Sampling interval (in seconds) that will be used if no proprer time vector is available in the log files.")
    parser.add_argument(
        "-fnum", "--fignum", default=DEFAULT_FIGNUM,
        help="Number or name of the figure.")
    parser.add_argument(
        "-tcn", "--time_col_name", type=str, default=None,
        help="Name of the relevant Time column in the log file. NB: no proper column in s-tui logs.")
    parser.add_argument(
        "-ucn", "--util_col_name", type=str, default=None,
        help="Name of the relevant Util. column (e.g., 'Util:Avg')' in the log file. Use the string 'none' to prevent looking for such data.")
    parser.add_argument(
        "-fcn", "--freq_col_name", type=str, default=None,
        help="Name of the relevant Frequency column (e.g., 'Frequency:Avg') in the log file Use the string 'none' to prevent looking for such data.")
    parser.add_argument(
        "-pcn", "--power_col_name", type=str, default=None,
        help="Name of the relevant power column (e.g., 'Power:package-0,0') in the log file. Use the string 'none' to prevent looking for such data.")

    args = parser.parse_args()

    # Massaging default values for arguments related to the name of the
    # targeted columns in the log file, as they depends on the software
    # used to record the data.

    xdata_target_name = DEFAULT_TARGETED_CSV_XDATA_COLUMN_NAME[args.software]
    ydata_target_names = DEFAULT_TARGETED_CSV_YDATA_COLUMN_NAMES[args.software]

    if args.time_col_name is None:
        try:
            args.time_col_name = xdata_target_name
        except KeyError:
            print("No column name has been given for Time.")

    if args.power_col_name is None:
        try:
            args.power_col_name = ydata_target_names[POWER_KEY]
        except KeyError:
            print("No column name has been given for Power.")

    if args.util_col_name is None:
        try:
            args.util_col_name = ydata_target_names[UTILIZATION_KEY]
        except KeyError:
            print("No column name has been given for Utilization.")

    if args.freq_col_name is None:
        try:
            args.freq_col_name = ydata_target_names[FREQUENCY_KEY]
        except KeyError:
            print("No column name has been given for Frequency.")

    return args

# %%
if __name__ == "__main__":
    #plt.ion()  # one might uncomment if one uses the script in IPython

    args = parse_arguments()

    log_container = LogContainer(args.software,
                                 time_col_name=args.time_col_name,
                                 power_col_name=args.power_col_name,
                                 util_col_name=args.util_col_name,
                                 freq_col_name=args.freq_col_name)
    log_container.load(args.filepath, dt=args.delta_t)
    fig, axs = log_container.plot(fignum=args.fignum)

    plt.show() # one might comment if one uses the script in IPython
