"""LCLM training package.

Heavy training dependencies are imported only when ``LCLMTrainer`` is actually
requested, keeping utility modules usable in preprocessing/test environments.
"""

__all__ = ["LCLMTrainer"]


def __getattr__(name):
    if name == "LCLMTrainer":
        from .trainer import LCLMTrainer

        return LCLMTrainer
    raise AttributeError(name)
