# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom setup.py for platform-specific wheel tagging.

This file exists solely to customize the wheel platform tag. All package metadata
is defined in pyproject.toml.

The native FFI libraries (.so/.dylib/.dll) require specific platform tags that
respect MACOSX_DEPLOYMENT_TARGET and ARCHFLAGS environment variables set by
cibuildwheel, rather than using sysconfig.get_platform() which returns Python's
compile-time values.
"""

import os
import platform
import subprocess
import sys

import setuptools  # type: ignore
from setuptools.command.build_py import build_py as _build_py  # type: ignore
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # type: ignore


def get_platform_tag():
    """Get the wheel platform tag for the current/target platform."""
    if sys.platform == "darwin":
        # Get deployment target from environment (set by cibuildwheel) or fall back
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if not target:
            target = platform.mac_ver()[0]
            parts = target.split(".")
            target = f"{parts[0]}.{parts[1] if len(parts) > 1 else '0'}"

        version_tag = target.replace(".", "_")

        # Check ARCHFLAGS for cross-compilation (cibuildwheel sets this)
        archflags = os.environ.get("ARCHFLAGS", "")
        if "-arch arm64" in archflags:
            arch = "arm64"
        elif "-arch x86_64" in archflags:
            arch = "x86_64"
        else:
            arch = platform.machine()

        return f"macosx_{version_tag}_{arch}"
    elif sys.platform == "linux":
        return f"linux_{platform.machine()}"
    elif sys.platform == "win32":
        arch = platform.machine()
        if arch == "AMD64":
            arch = "amd64"
        return f"win_{arch}"
    else:
        return f"{platform.system().lower()}_{platform.machine()}"


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        self.plat_name = get_platform_tag()
        _bdist_wheel.finalize_options(self)


class build_py(_build_py):
    """Ensure the prebuilt FFI native library is in livekit/rtc/resources
    before setuptools gathers package_data.

    Upstream relies on cibuildwheel's ``before-build`` hook to invoke
    ``rust-sdks/download_ffi.py``. That only runs in the cibuildwheel
    flow. When installing from source (e.g. ``pip install .`` or
    ``uv add git+...#subdirectory=livekit-rtc``), no one invokes
    download_ffi, so the wheel ships without the FFI and imports fail
    at runtime.

    This hook closes that gap: if no platform-native FFI library is
    already in ``livekit/rtc/resources``, download it before packaging.
    Respects ``LIVEKIT_FFI_REPO_URL`` and ``LIVEKIT_FFI_VERSION`` env
    vars handled by download_ffi.py.
    """

    _ffi_extensions = (".so", ".dylib", ".dll")

    def run(self):  # type: ignore[override]
        here = os.path.abspath(os.path.dirname(__file__))
        resources = os.path.join(here, "livekit", "rtc", "resources")
        needs_download = True
        if os.path.isdir(resources):
            for name in os.listdir(resources):
                if name.endswith(self._ffi_extensions):
                    needs_download = False
                    break
        if needs_download:
            os.makedirs(resources, exist_ok=True)
            script = os.path.join(here, "rust-sdks", "download_ffi.py")
            if os.path.exists(script):
                print(
                    f"[livekit-rtc setup] FFI missing; downloading via {script}",
                    file=sys.stderr,
                )
                subprocess.check_call([sys.executable, script, "--output", resources])
            else:
                raise RuntimeError(
                    "livekit-rtc: no FFI binary in livekit/rtc/resources and "
                    f"rust-sdks/download_ffi.py not found at {script!r}. "
                    "Did you clone with --recurse-submodules?"
                )
        super().run()


setuptools.setup(
    cmdclass={
        "bdist_wheel": bdist_wheel,
        "build_py": build_py,
    },
)
