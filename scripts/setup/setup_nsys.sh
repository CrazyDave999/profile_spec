#!/usr/bin/env bash
set -euo pipefail
version=2026.5.1
package=nsightsystems-linux-cli-public-2026.5.1.161-3889610.deb
url="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_5/$package"
checksum=61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08
cache="$HOME/.cache/nsight-systems"
prefix="$HOME/.local/opt/nsight-systems/$version"
mkdir -p "$cache" "$prefix" "$HOME/.local/bin"
if [[ ! -f "$cache/$package" ]]; then
  curl -fL --retry 3 --output "$cache/$package" "$url"
fi
printf '%s  %s\n' "$checksum" "$cache/$package" | sha256sum --check
dpkg-deb -x "$cache/$package" "$prefix"
target="$prefix/opt/nvidia/nsight-systems-cli/$version/target-linux-x64/nsys"
if [[ -e "$HOME/.local/bin/nsys" && ! -L "$HOME/.local/bin/nsys" ]]; then
  echo "Existing ~/.local/bin/nsys is not a symlink; installed CLI is at $target" >&2
else
  ln -sfn "$target" "$HOME/.local/bin/nsys"
fi
"$target" --version
"$target" status --environment
