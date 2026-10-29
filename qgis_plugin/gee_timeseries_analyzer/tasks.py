"""QgsTask wrapper around gee_client.extract_time_series.

Unlike the DWD plugin's few-second HTTP fetches, a GEE extraction (one
reduceRegions over every image of up to four collections) can take a minute
for a year of data over many polygons, so it runs in QGIS's task manager
instead of blocking the UI. Results are handed back on the main thread via
finished() -> on_done(task, ok)."""
from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from . import gee_client as c


class ExtractTask(QgsTask):
    progressText = pyqtSignal(str)

    def __init__(self, params, on_done):
        super().__init__("GEE time series extraction", QgsTask.CanCancel)
        self.params = params
        self.on_done = on_done
        self.df = None
        self.errors = {}
        self.exception = None

    def run(self):
        try:
            self.df, self.errors = c.extract_time_series(
                progress_cb=self._progress, cancel_cb=self.isCanceled, **self.params)
        except Exception as e:
            self.exception = e
            return False
        return not self.isCanceled()

    def _progress(self, i, n, msg):
        self.setProgress(100.0 * i / max(n, 1))
        self.progressText.emit(f"{msg} ({i + 1}/{n}) …")

    def finished(self, result):
        self.on_done(self, result)


class CallTask(QgsTask):
    """Runs fn(progress_cb=..., cancel_cb=..., **kwargs) in the task manager
    (the classification runs of classification.py); on_done(task, ok) on
    the main thread with task.result / task.exception."""
    progressText = pyqtSignal(str)

    def __init__(self, description, fn, kwargs, on_done):
        super().__init__(description, QgsTask.CanCancel)
        self.fn = fn
        self.kwargs = kwargs
        self.on_done = on_done
        self.result = None
        self.exception = None

    def run(self):
        try:
            self.result = self.fn(progress_cb=self._progress, cancel_cb=self.isCanceled, **self.kwargs)
        except InterruptedError:
            return False
        except Exception as e:
            self.exception = e
            return False
        return not self.isCanceled()

    def _progress(self, i, n, msg):
        self.setProgress(100.0 * i / max(n, 1))
        self.progressText.emit(f"{msg} ({i + 1}/{n}) …")

    def finished(self, result):
        self.on_done(self, result)
