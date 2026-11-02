"""Compatible installed-command adapters for packaged implementations."""

from __future__ import annotations


def run_video() -> int | None:
    from .commands.tracking.run_video import main

    return main()


def build_relative() -> int | None:
    from .commands.calibration.build_relative_trajectory import main

    return main()


def record_d455() -> int:
    from .d455 import main

    return main()


def record_t265() -> int:
    from .t265 import main

    return main()


def live_dashboard() -> int | None:
    from .commands.tracking.live_dashboard import main

    return main()
