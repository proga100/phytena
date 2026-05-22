"""Strategy pattern for dataset downloaders.

Each Downloader knows how to fetch from one source (Kaggle, GitHub, ...).
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from growz_eval.logging_utils import get_logger

log = get_logger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download fails for a recoverable reason."""


class AuthError(RuntimeError):
    """Raised when downloader auth is missing/invalid."""


class Downloader(ABC):
    """Fetches a single dataset into ``out_dir``."""

    @abstractmethod
    def ensure_ready(self) -> None:
        """Raise AuthError or DownloadError if the downloader can't run."""

    @abstractmethod
    def fetch(self, locator: str, out_dir: Path) -> None:
        """Download ``locator`` into ``out_dir``."""


class KaggleDownloader(Downloader):
    """Uses the ``kaggle`` CLI. Auth via ~/.kaggle/access_token or kaggle.json."""

    def ensure_ready(self) -> None:
        if shutil.which("kaggle") is None:
            raise DownloadError("Kaggle CLI not found. Run: pip install kaggle")

        kaggle_dir = Path.home() / ".kaggle"
        has_json = (kaggle_dir / "kaggle.json").exists()
        has_token = (kaggle_dir / "access_token").exists()
        if not (has_json or has_token):
            raise AuthError(
                f"No Kaggle credentials. Place kaggle.json or access_token in {kaggle_dir}."
            )

    def fetch(self, locator: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("  downloading kaggle:%s", locator)
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", locator,
                 "-p", str(out_dir), "--unzip"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(
                f"kaggle download failed for {locator}: {exc.stderr[:200]}"
            ) from exc
        log.info("  done: %s", locator)


class GitHubDownloader(Downloader):
    """Shallow-clones a public GitHub repo."""

    def ensure_ready(self) -> None:
        if shutil.which("git") is None:
            raise DownloadError("git not found on PATH")

    def fetch(self, locator: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("  cloning %s", locator)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", locator, str(out_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(
                f"git clone failed for {locator}: {exc.stderr[:200]}"
            ) from exc
        log.info("  done: %s", locator)


def downloader_for(source: str) -> Downloader:
    if source == "kaggle":
        return KaggleDownloader()
    if source == "github":
        return GitHubDownloader()
    raise ValueError(f"Unknown source: {source!r}")
