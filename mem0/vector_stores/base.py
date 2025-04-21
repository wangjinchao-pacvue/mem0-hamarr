from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class FilterBase(ABC):
    """Base class for vector store filters."""

    @abstractmethod
    def build(self) -> Any:
        """Build the filter into the format expected by the vector store."""
        pass

    def match(self, key: str, value: Any) -> "FilterBase":
        """Add a match condition. Override if supported."""
        raise NotImplementedError("match operation not supported by this vector store")

    def match_any(self, key: str, values: List[Any]) -> "FilterBase":
        """Add a match any condition. Override if supported."""
        raise NotImplementedError("match_any operation not supported by this vector store")

    def match_except(self, key: str, values: List[Any]) -> "FilterBase":
        """Add a match except condition. Override if supported."""
        raise NotImplementedError("match_except operation not supported by this vector store")

    def range(self, key: str, gt: Optional[float] = None, gte: Optional[float] = None,
             lt: Optional[float] = None, lte: Optional[float] = None) -> "FilterBase":
        """Add a range condition. Override if supported."""
        raise NotImplementedError("range operation not supported by this vector store")

    def has_id(self, ids: List[int]) -> "FilterBase":
        """Add a has id condition. Override if supported."""
        raise NotImplementedError("has_id operation not supported by this vector store")

    def has_vector(self, vector_name: str) -> "FilterBase":
        """Add a has vector condition. Override if supported."""
        raise NotImplementedError("has_vector operation not supported by this vector store")

    def is_empty(self, key: str) -> "FilterBase":
        """Add an is empty condition. Override if supported."""
        raise NotImplementedError("is_empty operation not supported by this vector store")

    def is_null(self, key: str) -> "FilterBase":
        """Add an is null condition. Override if supported."""
        raise NotImplementedError("is_null operation not supported by this vector store")

    def values_count(self, key: str, gt: Optional[int] = None, gte: Optional[int] = None,
                    lt: Optional[int] = None, lte: Optional[int] = None) -> "FilterBase":
        """Add a values count condition. Override if supported."""
        raise NotImplementedError("values_count operation not supported by this vector store")

    def must(self, condition: Any) -> "FilterBase":
        """Add a condition that must be satisfied. Override if supported."""
        raise NotImplementedError("must operation not supported by this vector store")

    def should(self, condition: Any) -> "FilterBase":
        """Add a condition that should be satisfied. Override if supported."""
        raise NotImplementedError("should operation not supported by this vector store")

    def must_not(self, condition: Any) -> "FilterBase":
        """Add a condition that must not be satisfied. Override if supported."""
        raise NotImplementedError("must_not operation not supported by this vector store")


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    def search(self, query, vectors, limit=5, filters: Optional[Union[Dict, FilterBase]] = None):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""
        pass

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""
        pass

    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""
        pass

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""
        pass

    @abstractmethod
    def list(self, filters: Optional[Union[Dict, FilterBase]] = None, limit: int = 100):
        """List all vectors in a collection."""
        pass

    def _create_filter(self, filters: Optional[Union[Dict, FilterBase]]) -> Any:
        """
        Create a filter object from the provided filters.
        This is an optional method that vector stores can override if they support complex filtering.
        By default, it returns None, indicating that the vector store should handle filters in its own way.

        Args:
            filters (Union[dict, FilterBase], optional): Filters to apply.

        Returns:
            Any: The created filter object, or None if not supported.
        """
        return None
