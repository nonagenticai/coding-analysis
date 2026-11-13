"""System models for package management and OS-level operations.

TODO: Implement full system models.
"""

from __future__ import annotations

import enum


class PackageManager(enum.StrEnum):
    """Supported system package managers."""

    APT = "apt"
    YUM = "yum"
    DNF = "dnf"
    BREW = "brew"
    PACMAN = "pacman"
    APK = "apk"
