from collections.abc import Iterable


class CubeCallback:
    """Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks."""

    def on_run_start(self):
        """Called when the run begins."""

    def on_run_end(self):
        """Called when the run ends."""


class CubeCallbackList:
    """A container to store and invoke `CubeCallback`s."""

    def __init__(self, callbacks: CubeCallback | Iterable[CubeCallback] | None):
        if callbacks is None:
            self._callbacks = []
        elif isinstance(callbacks, CubeCallback):
            self._callbacks = [callbacks]
        elif isinstance(callbacks, Iterable):
            self._callbacks = callbacks
        else:
            raise ValueError(f"Unsupported callback type: {type(callbacks)}")

    def on_run_start(self):
        for callback in self._callbacks:
            callback.on_run_start()

    def on_run_end(self):
        for callback in self._callbacks:
            callback.on_run_end()
