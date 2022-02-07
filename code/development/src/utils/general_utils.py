"""
Utils that are quite general.
"""

import hashlib
import os
import time
import warnings
from secrets import token_hex
from typing import Any, List, Optional, Tuple

import numpy as np


def ensure_dir_exists(path: str) -> None:
    path = os.path.abspath(os.path.realpath(path))
    if not os.path.exists(path):
        os.makedirs(path)


def outpath(path: str) -> str:
    path = os.path.abspath(os.path.realpath(path))
    ensure_dir_exists(os.path.dirname(path))
    return path


def blend(*images: np.ndarray) -> np.ndarray:
    out = np.zeros(images[0].shape)
    for image in images:
        assert image.ndim == 3
        assert image.shape[-1] == 4
        new_alpha = image[:, :, 3] + out[:, :, 3] * (1 - image[:, :, 3])
        for i in range(3):
            out[:, :, i] = image[:, :, i] * image[:, :, 3] + out[:, :, i] * out[:, :, 3] * (1 - image[:, :, 3])
        out[:, :, 3] = new_alpha
    return out


def grayscale_to_rgba(grayscale: np.ndarray) -> np.ndarray:
    return np.stack([grayscale] * 3 + [np.ones(grayscale.shape)], axis=-1)


def mask_to_image(mask: np.ndarray, color: Tuple[float, float, float, float]) -> np.ndarray:
    assert set(np.unique(mask)).issubset({0, 1})
    out = np.ones(mask.shape + (4,))
    for i in range(4):
        out[..., i] = np.where(mask == 1, color[i], 0)
    return out


def get_eval_folder_path(
    model_name: str, run_name: str, use_shared_folder: bool = False, test_database: bool = False
) -> str:
    """Return the path to the evaluation folder that corespond to a given run and model.

    The evaluation folder is the place to store predictions and evaluations of
    the run in general, such as the results of metrics analysis.
    """
    base_folder = "" if use_shared_folder else "personal"
    eval_folder = os.path.join(base_folder, "eval", model_name, run_name)
    if test_database:
        eval_folder += "/test_database"
    return eval_folder


def make_sync_request(relative_path: str, safe_sync: bool = True) -> None:
    """Create a sync request to a dirsyncd daemon.

    If the environment variable DIRSYNCC_REQ_DIR (which stands for dirsync
    client request directory) is set, issue a request to move/copy the data in
    relative_path to another folder.

    safe_sync=False removes the restriction to only move files within the personal
    folder, only use this option if you know what you're doing.

    This is the client function for the daemon dyrsincd, see
    https://github.com/rlschuller/dirsyncd for more details.
    """

    try:
        DIRSYNCC_REQ_DIR = os.environ["DIRSYNCC_REQ_DIR"]
    except KeyError:
        print("Warning: DIRSYNCC_REQ_DIR isn't set, make_sync_request will do nothing.")
        return

    if safe_sync and "personal" not in os.path.abspath(os.path.join(os.environ["PYTHONPATH"], relative_path)):
        print(
            f"Warning: since safe_sync==True, make_sync_request can only sync files within personal folder. {relative_path} isn't inside a personal folder. Only use safe_sync=False if you know what you're doing..."
        )
        return

    tmp_filename = "error_" + token_hex(32)
    with open(os.path.join(DIRSYNCC_REQ_DIR, tmp_filename), "w") as f:
        f.write(relative_path)

    print(f"Making sync request to move {relative_path} in folder {DIRSYNCC_REQ_DIR}")
    os.rename(os.path.join(DIRSYNCC_REQ_DIR, tmp_filename), os.path.join(DIRSYNCC_REQ_DIR, "request_" + token_hex(32)))


def encode_string(string: str) -> int:
    return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % 10 ** 8


def normalize_image(image: np.ndarray, low: int = 0, high: int = 255) -> np.ndarray:
    """Normalize the data to a certain range. Default: [0-255]"""
    image = np.asarray(image)
    out = np.interp(image, (image.min(), image.max()), (low, high))
    if (low, high) == (0, 1):
        out = out.astype(np.float64)
    else:
        out = out.astype(np.uint8)
    return out


class _TicToc(object):
    """
    Author: Hector Sanchez
    Date: 2018-07-26
    Description: Class that allows you to do 'tic toc' to your code.
    This class was based on https://github.com/hector-sab/ttictoc, which is
    distributed under the MIT license. It prints time information between
    successive tic() and toc() calls.
    Example:
        from src.utils.general_utils import tic,toc
        tic()
        tic()
        toc()
        toc()
    """

    def __init__(
        self,
        name: str = "",
        method: Any = "time",
        nested: bool = False,
        print_toc: bool = True,
    ) -> None:
        """
        Args:
            name (str): Just informative, not needed
            method (int|str|ftn|clss): Still trying to understand the default
                options. 'time' uses the 'real wold' clock, while the other
                two use the cpu clock. To use your own method,
                do it through this argument
                Valid int values:
                    0: time.time | 1: time.perf_counter | 2: time.proces_time
                    3: time.time_ns | 4: time.perf_counter_ns
                    5: time.proces_time_ns
                Valid str values:
                  'time': time.time | 'perf_counter': time.perf_counter
                  'process_time': time.proces_time | 'time_ns': time.time_ns
                  'perf_counter_ns': time.perf_counter_ns
                  'proces_time_ns': time.proces_time_ns
                Others:
                  Whatever you want to use as time.time
            nested (bool): Allows to do tic toc with nested with a
                single object. If True, you can put several tics using the
                same object, and each toc will correspond to the respective tic.
                If False, it will only register one single tic, and
                return the respective elapsed time of the future tocs.
            print_toc (bool): Indicates if the toc method will print
                the elapsed time or not.
        """
        self.name = name
        self.nested = nested
        self.tstart: Any[List, None] = None
        if self.nested:
            self.set_nested(True)

        self._print_toc = print_toc

        self._int2strl = [
            "time",
            "perf_counter",
            "process_time",
            "time_ns",
            "perf_counter_ns",
            "process_time_ns",
        ]
        self._str2fn = {
            "time": [time.time, "s"],
            "perf_counter": [time.perf_counter, "s"],
            "process_time": [time.process_time, "s"],
            "time_ns": [time.time_ns, "ns"],
            "perf_counter_ns": [time.perf_counter_ns, "ns"],
            "process_time_ns": [time.process_time_ns, "ns"],
        }

        if type(method) is not int and type(method) is not str:
            self._get_time = method

        if type(method) is int and method < len(self._int2strl):
            method = self._int2strl[method]
        elif type(method) is int and method > len(self._int2strl):
            method = "time"

        if type(method) is str and method in self._str2fn:
            self._get_time = self._str2fn[method][0]
            self._measure = self._str2fn[method][1]
        elif type(method) is str and method not in self._str2fn:
            self._get_time = self._str2fn["time"][0]
            self._measure = self._str2fn["time"][1]

    def _print_elapsed(self) -> None:
        """
        Prints the elapsed time
        """
        if self.name != "":
            name = "[{}] ".format(self.name)
        else:
            name = self.name
        print("-{0}elapsed time: {1:.3g} ({2})".format(name, self.elapsed, self._measure))

    def tic(self) -> None:
        """
        Defines the start of the timing.
        """
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def toc(self, print_elapsed: Optional[bool] = None) -> None:
        """
        Defines the end of the timing.
        """
        self.tend = self._get_time()
        if self.nested:
            if len(self.tstart) > 0:
                self.elapsed = self.tend - self.tstart.pop()
            else:
                self.elapsed = None
        else:
            if self.tstart:
                self.elapsed = self.tend - self.tstart
            else:
                self.elapsed = None

        if print_elapsed is None:
            if self._print_toc:
                self._print_elapsed()
        else:
            if print_elapsed:
                self._print_elapsed()

        # return(self.elapsed)

    def set_print_toc(self, set_print: bool) -> None:
        """
        Indicate if you want the timed time printed out or not.
        Args:
          set_print (bool): If True, a message with the elapsed time
            will be printed.
        """
        if type(set_print) is bool:
            self._print_toc = set_print
        else:
            warnings.warn(
                "Parameter 'set_print' not boolean. " "Ignoring the command.",
                Warning,
            )

    def set_nested(self, nested: bool) -> None:
        """
        Sets the nested functionality.
        """
        # Assert that the input is a boolean
        if type(nested) is bool:
            # Check if the request is actually changing the
            # behaviour of the nested tictoc
            if nested != self.nested:
                self.nested = nested

                if self.nested:
                    self.tstart = []
                else:
                    self.tstart = None
        else:
            warnings.warn(
                "Parameter 'nested' not boolean. " "Ignoring the command.",
                Warning,
            )


class TicToc(_TicToc):
    def tic(self, nested: bool = True) -> None:
        """
        Defines the start of the timing.
        """
        if nested:
            self.set_nested(True)

        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()


__TICTOC_8320947502983745 = TicToc()
tic = __TICTOC_8320947502983745.tic
toc = __TICTOC_8320947502983745.toc


class FormatText:
    """Print with colors/formatation
    examples:
        print(FormatText.BOLD + 'Hello World !' + FormatText.END)
        print(FormatText.FORMATED_FAIL)
    """

    TITLE = "\x1b[1;33m"
    BOLD = "\x1b[1m"
    WARNING = "\x1b[1;30;43m"
    ERROR = "\x1b[1;30;41m"
    FAIL = "\x1b[1;31m"
    END = "\x1b[0m"
    OK = "\x1b[1;32m"
    FORMATED_OK = "[  \x1b[1;32mOK" + END + "  ]"
    FORMATED_FAIL = "[ \x1b[1;31mFAIL" + END + " ]"
    FORMATED_WARNING = WARNING + "[ WARNING ]" + END
    FORMATED_ERROR = ERROR + "[ ERROR ]" + END


def print_title(printable: Any) -> None:
    print(FormatText.TITLE + str(printable) + FormatText.END)
