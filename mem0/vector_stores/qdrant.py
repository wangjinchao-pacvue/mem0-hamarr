import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchExcept,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    ValuesCount,
    VectorParams,
    GeoBoundingBox,
    GeoPoint,
    GeoRadius,
    GeoPolygon,
    PayloadField,
)

from mem0.vector_stores.base import FilterBase, VectorStoreBase

logger = logging.getLogger(__name__)


class QdrantFilter(FilterBase):
    """Qdrant-specific implementation of the filter builder."""

    def __init__(self):
        self._conditions = []
        self._must = []
        self._should = []
        self._must_not = []

    def build(self) -> Filter:
        """Build the final filter."""
        if self._conditions:
            if self._must or self._should or self._must_not:
                raise ValueError("Cannot mix conditions with must/should/must_not clauses")
            return Filter(must=self._conditions)

        return Filter(
            must=self._must if self._must else None,
            should=self._should if self._should else None,
            must_not=self._must_not if self._must_not else None
        )

    def match(self, key: str, value: Any) -> "QdrantFilter":
        """Add a match condition."""
        self._conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return self

    def match_any(self, key: str, values: List[Any]) -> "QdrantFilter":
        """Add a match any condition."""
        self._conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
        return self

    def match_except(self, key_name: str, values: List[Any]) -> "QdrantFilter":
        """Add a match except condition."""
        self._conditions.append(FieldCondition(key=key_name, match=MatchExcept(**{"except": values})))
        return self

    def range(self, key_name: str, gt: Optional[float] = None, gte: Optional[float] = None,
             lt: Optional[float] = None, lte: Optional[float] = None) -> "QdrantFilter":
        """Add a range condition."""
        self._conditions.append(FieldCondition(key=key_name, range=Range(gt=gt, gte=gte, lt=lt, lte=lte)))
        return self

    def has_id(self, ids: List[int]) -> "QdrantFilter":
        """Add a has id condition."""
        self._conditions.append(HasIdCondition(has_id=ids))
        return self

    def has_vector(self, vector_name: str) -> "QdrantFilter":
        """Add a has vector condition."""
        self._conditions.append(HasVectorCondition(has_vector=vector_name))
        return self

    def is_empty(self, key_name: str) -> "QdrantFilter":
        """Add an is empty condition."""
        self._conditions.append(IsEmptyCondition(is_empty=PayloadField(key=key_name)))
        return self

    def is_null(self, key_name: str) -> "QdrantFilter":
        """Add an is null condition."""
        self._conditions.append(IsNullCondition(is_null=PayloadField(key=key_name)))
        return self

    def values_count(self, key: str, gt: Optional[int] = None, gte: Optional[int] = None,
                    lt: Optional[int] = None, lte: Optional[int] = None) -> "QdrantFilter":
        """Add a values count condition."""
        self._conditions.append(FieldCondition(key=key, values_count=ValuesCount(gt=gt, gte=gte, lt=lt, lte=lte)))
        return self

    def geo_bounding_box(self, key: str, bottom_right: Dict[str, float], top_left: Dict[str, float]) -> "QdrantFilter":
        """Add a geo bounding box condition."""
        self._conditions.append(FieldCondition(
            key=key,
            geo_bounding_box=GeoBoundingBox(
                bottom_right=GeoPoint(**bottom_right),
                top_left=GeoPoint(**top_left)
            )
        ))
        return self

    def geo_radius(self, key: str, center: Dict[str, float], radius: float) -> "QdrantFilter":
        """Add a geo radius condition."""
        self._conditions.append(FieldCondition(
            key=key,
            geo_radius=GeoRadius(center=GeoPoint(**center), radius=radius)
        ))
        return self

    def geo_polygon(self, key: str, exterior: List[Dict[str, float]], interiors: Optional[List[List[Dict[str, float]]]] = None) -> "QdrantFilter":
        """Add a geo polygon condition."""
        self._conditions.append(FieldCondition(
            key=key,
            geo_polygon=GeoPolygon(
                exterior={"points": [GeoPoint(**point) for point in exterior]},
                interiors=[{"points": [GeoPoint(**point) for point in interior]} for interior in interiors] if interiors else None
            )
        ))
        return self

    def must(self, condition: Any) -> "QdrantFilter":
        """Add a condition that must be satisfied."""
        if isinstance(condition, QdrantFilter):
            self._must.extend(condition._conditions)
        else:
            self._must.append(condition)
        return self

    def should(self, condition: Any) -> "QdrantFilter":
        """Add a condition that should be satisfied."""
        if isinstance(condition, QdrantFilter):
            self._should.extend(condition._conditions)
        else:
            self._should.append(condition)
        return self

    def must_not(self, condition: Any) -> "QdrantFilter":
        """Add a condition that must not be satisfied."""
        if isinstance(condition, QdrantFilter):
            self._must_not.extend(condition._conditions)
        else:
            self._must_not.append(condition)
        return self


class Qdrant(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: QdrantClient | None = None,
        host: str = None,
        port: int = None,
        path: str = None,
        url: str = None,
        api_key: str = None,
        on_disk: bool = False,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
        """
        if client:
            self.client = client
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port
            if not params:
                params["path"] = path
                if not on_disk:
                    if os.path.exists(path) and os.path.isdir(path):
                        shutil.rmtree(path)

            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.create_col(embedding_model_dims, on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
        )

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        points = [
            PointStruct(
                id=idx if ids is None else ids[idx],
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _create_filter(self, filters: Optional[Union[Dict, FilterBase]]) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (Union[dict, FilterBase], optional): Filters to apply. Can be a dict for backward compatibility
                or a FilterBase instance for advanced filtering.

        Returns:
            Filter: The created Filter object.
        """
        if filters is None:
            return None

        if isinstance(filters, FilterBase):
            return filters.build()

        # Backward compatibility for simple dict filters
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                if "gte" in value and "lte" in value:
                    conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
                elif "gt" in value or "gte" in value or "lt" in value or "lte" in value:
                    conditions.append(FieldCondition(key=key, range=Range(
                        gt=value.get("gt"),
                        gte=value.get("gte"),
                        lt=value.get("lt"),
                        lte=value.get("lte")
                    )))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            vectors (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            query_filter=query_filter,
            limit=limit,
        )
        return hits.points

    def delete(self, vector_id: int):
        """
        Delete a vector by ID.

        Args:
            vector_id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result
