from typing import Any


class _SharedConfig:
    """
    Implement a shared config using a singleton pattern.
    Note that this way is not multithread-safe, but considering that the configuration process
    is single-threaded (at least for now), this easy way is adopted.

    Usage:
    - In the root config getter: import `shared_config` and use `shared_config.set(name, value)`
      to set all the values needed to be shared among config components.
    - In the config components' getters: import `shared_config` and use `shared_config.get(name)`
      to get the value.
    """

    _data = {}

    def set(self, name: str, value: Any):
        self._data[name] = value

    def get(self, name: str) -> Any:
        if name not in self._data:
            raise ValueError(f"There is no configuration named {name}.")
        return self._data[name]


# Only expose an instance
shared_config = _SharedConfig()
