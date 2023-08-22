import glob
import importlib
import os



class SubmoduleLoader(object):
    """Load submodules from the given ``package``

    :param package: package to load
    """

    def __init__(self, package):
        self._package = package
        self._package_object = importlib.import_module(self._package)
        if (
            getattr(self._package_object, "__file__", None) is not None
        ):  # pragma: no cover
            self._package_path = os.path.dirname(self._package_object.__file__)
        else:
            self._package_path = next(iter(self._package_object.__path__))

    def iter_modules(self, pattern=None):
        """Returns a list of submodules in the ``package`` and import them

        :param pattern: pattern to match modules, default *.py
        """
        for mod in self._find_modules(pattern):
            # Skip the private module, such as __init__.py
            if mod.startswith("_"):  # pragma: no cover
                continue

            name = mod.split(".")[0]
            yield importlib.import_module("." + name, self._package)

        for pkg in self._iter_sub_packages():
            sub_loader = SubmoduleLoader("{}.{}".format(self._package, pkg))
            for m in sub_loader.iter_modules(pattern):
                yield m

    def _find_modules(self, pattern=None):
        """Returns a list of submodules in the ``package``."""

        if pattern is None:  # pragma: no cover
            pattern = "*.py"

        modules_in_str = glob.glob1(self._package_path, pattern)
        return modules_in_str

    def _iter_sub_packages(self):
        "Iterates sub packages in the ``package``."
        for item in os.listdir(self._package_path):
            if self._is_sub_package(item):
                yield item

    def _is_dir(self, submodule):
        """Returns true if the ``submodule`` in the ``pacakge`` is a
        directory.
        """

        return os.path.isdir(os.path.join(self._package_path, submodule))

    def _is_sub_package(self, submodule):
        """Returns True if the ``submodule`` in the ``package`` is a sub
        package.
        """
        return self._is_dir(submodule) and os.path.exists(
            os.path.join(self._package_path, submodule, "__init__.py")
        )


SETTINGS_MODULE_KEY = "FLAG_SETTINGS_MODULE"


def get_settings_module():
    if SETTINGS_MODULE_KEY not in os.environ:  # pragma: no cover
        raise RuntimeError(
            "please set environment variable %s" % SETTINGS_MODULE_KEY
        )
    return importlib.import_module(os.environ[SETTINGS_MODULE_KEY])
