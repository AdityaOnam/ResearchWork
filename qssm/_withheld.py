"""
_withheld — marker for implementation that is not part of the public release.

The QSSM method is under review for publication. The files listed in WITHHELD.md keep
their public interface (module, class and function names, signatures, docstrings) so the
repository structure, imports and documentation stay intact, but their bodies raise
`WithheldError`. The full implementation will be released on publication.
"""

WITHHELD_MSG = ("This component of QSSM is withheld under lab confidentiality until "
                "publication (see WITHHELD.md).")


class WithheldError(NotImplementedError):
    def __init__(self, what: str = ""):
        super().__init__(f"{what}: {WITHHELD_MSG}" if what else WITHHELD_MSG)
