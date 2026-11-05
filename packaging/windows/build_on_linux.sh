#!/usr/bin/env bash
# Build the Windows bundle + installer of "AI Remove - thsky90" from Linux.
#   needs: curl, unzip, pip, x86_64-w64-mingw32-gcc/windres, makensis
#   usage: packaging/windows/build_on_linux.sh [cache_dir] [out_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CACHE="${1:-$ROOT/.build-cache}"
OUT="${2:-$ROOT/dist}"
# absolute paths: later steps cd into sub-folders (relative "dist" broke the zip step on CI)
mkdir -p "$CACHE" "$OUT"
CACHE="$(cd "$CACHE" && pwd)"
OUT="$(cd "$OUT" && pwd)"
PYVER="3.12.10"
APP="AI Remove - thsky90"
VERSION="$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('$ROOT/airemove/__init__.py').read()).group(1))")"
STAGE="$OUT/stage/$APP"
mkdir -p "$CACHE/wheels" "$OUT"
rm -rf "$OUT/stage"
mkdir -p "$STAGE"

echo "== 1. Windows Python $PYVER (NuGet package)"
if [ ! -f "$CACHE/python.$PYVER.nupkg" ]; then
  curl -sSL -o "$CACHE/python.$PYVER.nupkg" "https://api.nuget.org/v3-flatcontainer/python/$PYVER/python.$PYVER.nupkg"
fi
rm -rf "$CACHE/pyroot" && mkdir -p "$CACHE/pyroot"
unzip -q "$CACHE/python.$PYVER.nupkg" 'tools/*' -d "$CACHE/pyroot"
cp -r "$CACHE/pyroot/tools" "$STAGE/runtime"
rm -rf "$STAGE/runtime/include" "$STAGE/runtime/libs"
for d in test idlelib tkinter turtledemo ensurepip lib2to3 pydoc_data unittest/test; do
  rm -rf "$STAGE/runtime/Lib/$d"
done
find "$STAGE/runtime" -name "__pycache__" -type d -prune -exec rm -rf {} +
# isolated interpreter: ignore PYTHONPATH/registry of any other Python on the PC
cat > "$STAGE/runtime/python312._pth" <<'EOF'
.
Lib
Lib\site-packages
DLLs
..\app
import site
EOF

echo "== 2. Python packages (win_amd64 wheels)"
pip download -q --only-binary=:all: --platform win_amd64 --python-version 3.12 --implementation cp \
  -d "$CACHE/wheels" -r "$ROOT/requirements.txt"
pip download -q --only-binary=:all: --platform win_amd64 --no-deps -d "$CACHE/wheels" imageio-ffmpeg==0.8.1
SP="$STAGE/runtime/Lib/site-packages"
mkdir -p "$SP"
for w in "$CACHE"/wheels/*.whl; do
  case "$(basename "$w")" in imageio_ffmpeg*) continue;; esac
  unzip -q -o "$w" -d "$SP"
done
# move *.data/purelib|platlib contents into site-packages (wheel spec)
for d in "$SP"/*.data; do
  [ -d "$d" ] || continue
  for sub in purelib platlib; do
    [ -d "$d/$sub" ] && cp -r "$d/$sub/." "$SP/"
  done
  rm -rf "$d"
done
# trim Qt: the app only needs QtCore/QtGui/QtWidgets
Q="$SP/PySide6"
rm -rf "$Q/include" "$Q/typesystems" "$Q/glue" "$Q/examples" "$Q/qml" "$Q/translations" \
       "$Q/resources" "$Q/metatypes" "$Q/doc" "$Q/scripts" "$Q/support" "$Q/Qt/lib/cmake"
find "$Q" -maxdepth 1 -type f \( -name "*.exe" -o -name "opengl32sw.dll" -o -name "d3dcompiler*.dll" \) -delete
# whitelist: keep only Core/Gui/Widgets
for f in "$Q"/Qt6*.dll; do
  case "$(basename "$f")" in Qt6Core.dll|Qt6Gui.dll|Qt6Widgets.dll) ;; *) rm -f "$f";; esac
done
for f in "$Q"/Qt*.pyd "$Q"/Qt*.pyi; do
  case "$(basename "$f")" in QtCore.pyd|QtGui.pyd|QtWidgets.pyd) ;; *) rm -f "$f";; esac
done
if [ -d "$Q/plugins" ]; then
  for d in "$Q"/plugins/*; do
    case "$(basename "$d")" in platforms|styles|imageformats) ;; *) rm -rf "$d";; esac
  done
  find "$Q/plugins/imageformats" -type f ! -name "qico*" ! -name "qjpeg*" -delete 2>/dev/null || true
  find "$Q/plugins/platforms" -type f ! -name "qwindows*" ! -name "qoffscreen*" ! -name "qminimal*" -delete 2>/dev/null || true
fi
rm -f "$SP"/cv2/opencv_videoio_ffmpeg*.dll
find "$SP" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$SP" -name "*.pyi" -delete

echo "== 3. ffmpeg.exe"
mkdir -p "$STAGE/ffmpeg" "$CACHE/ff"
unzip -q -o "$CACHE"/wheels/imageio_ffmpeg-*-win_amd64.whl 'imageio_ffmpeg/binaries/*' -d "$CACHE/ff"
cp "$(ls "$CACHE"/ff/imageio_ffmpeg/binaries/ffmpeg-win*.exe | head -1)" "$STAGE/ffmpeg/ffmpeg.exe"

echo "== 4. application"
mkdir -p "$STAGE/app"
cp -r "$ROOT/airemove" "$STAGE/app/"
cp "$ROOT/main.py" "$STAGE/app/"
find "$STAGE/app" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$ROOT/LICENSE" "$STAGE/LICENSE.txt"
cp "$ROOT/packaging/windows/README-Windows.txt" "$STAGE/HUONG-DAN.txt"

echo "== 5. launcher exe"
B="$OUT/build-launcher"
rm -rf "$B" && mkdir -p "$B"
cp "$HERE/launcher.c" "$B/"
VCOMMA="$(echo "$VERSION" | tr '.' ',')"
sed -e "s/1,0,0,0/$VCOMMA,0/g" -e "s/\"1\.0\.0\"/\"$VERSION\"/g" "$HERE/launcher.rc" > "$B/launcher.rc"
cp "$ROOT/airemove/assets/app_icon.ico" "$B/"
( cd "$B" && x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher.res \
  && x86_64-w64-mingw32-gcc -O2 -s -municode -mwindows launcher.c launcher.res -o launcher.exe )
cp "$B/launcher.exe" "$STAGE/$APP.exe"

if [ "${SKIP_PACK:-0}" = "1" ]; then du -sh "$STAGE"; exit 0; fi

echo "== 6. portable zip"
( cd "$OUT/stage" && rm -f "$OUT/AI-Remove-thsky90-$VERSION-portable.zip" && \
  zip -q -r -9 "$OUT/AI-Remove-thsky90-$VERSION-portable.zip" "$APP" )

echo "== 7. installer"
cp "$ROOT/airemove/assets/app_icon.ico" "$OUT/stage/app_icon.ico"
cp "$HERE/installer.nsi" "$OUT/stage/installer.nsi"
( cd "$OUT/stage" && makensis -V2 -DVERSION="$VERSION" -DSTAGE="$APP" \
  -DOUTFILE="$OUT/AI-Remove-thsky90-$VERSION-Setup.exe" installer.nsi )
ls -la "$OUT"/*.exe "$OUT"/*.zip
du -sh "$STAGE"
