import glob
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
PYFLEX_ROOT = os.environ.get('PYFLEXROOT', os.path.join(ROOT, 'PyFlex'))

if not os.path.isdir(PYFLEX_ROOT):
    raise ModuleNotFoundError(
        "No local PyFlex root found. Set PYFLEXROOT to the directory containing PyFlex or build the local PyFlex package."
    )

os.environ.setdefault('PYFLEXROOT', PYFLEX_ROOT)

SEARCH_DIRS = [
    os.path.join(PYFLEX_ROOT, 'bindings', 'build'),
    os.path.join(PYFLEX_ROOT, 'bindings'),
    os.path.join(PYFLEX_ROOT, 'build'),
]

candidates = []
for search_dir in SEARCH_DIRS:
    if os.path.isdir(search_dir):
        candidates.extend(glob.glob(os.path.join(search_dir, 'pyflex*.so')))
        candidates.extend(glob.glob(os.path.join(search_dir, 'pyflex*.pyd')))

if not candidates:
    raise ModuleNotFoundError(
        f"Could not find a compiled pyflex binary in {SEARCH_DIRS}.\n"
        "Build PyFlex for your Python version and place the extension in PyFlex/bindings/build. "
        "If you are using Python 3.12, the existing binding may be for Python 3.9 and must be rebuilt."
    )

# Prefer a binary matching this Python version if available.
version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
version_matches = [p for p in candidates if version_tag in os.path.basename(p)]
module_path = version_matches[0] if version_matches else candidates[0]

spec = importlib.util.spec_from_file_location('pyflex', module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load pyflex extension from {module_path}")
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Exception as exc:
    raise ImportError(
        f"Failed to import local pyflex extension at {module_path}: {exc}\n"
        "If this is a Python version mismatch, rebuild PyFlex for the current interpreter."
    ) from exc

sys.modules['pyflex'] = module

# Expose the loaded pyflex module under the name pyflex.
pyflex = module
