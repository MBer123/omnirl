import ray


@ray.remote
class WeightStore:
    """
    A lightweight Ray actor for storing and serving model weight references.

    Args:
        weights_refs: Single-item list preserving the initial weight reference.
        version: Initial model version.
    """

    def __init__(self, weights_refs: list[ray.ObjectRef], version: int):
        self.version = version
        self.weights_ref = weights_refs[0]

    def set(self, weights_refs, version):
        """Update the nested weight reference and its version number."""
        self.version = version
        self.weights_ref = weights_refs[0]

    def get(self, old_version):
        """Return the shared weight reference if a newer version exists."""
        if self.version > old_version:
            return {"version": self.version, "weights_ref": self.weights_ref}
        return {"version": self.version, "weights_ref": None}
