import unittest
from unittest.mock import MagicMock
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PointIdsList,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchExcept,
    Range,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    ValuesCount,
    GeoBoundingBox,
    GeoRadius,
    GeoPolygon,
    Filter,
)
from mem0.vector_stores.qdrant import Qdrant, QdrantFilter


class TestQdrant(unittest.TestCase):
    def setUp(self):
        self.client_mock = MagicMock(spec=QdrantClient)
        self.qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=self.client_mock,
            path="test_path",
            on_disk=True,
        )

    def test_create_col(self):
        self.client_mock.get_collections.return_value = MagicMock(collections=[])

        self.qdrant.create_col(vector_size=128, on_disk=True)

        expected_config = VectorParams(size=128, distance=Distance.COSINE, on_disk=True)

        self.client_mock.create_collection.assert_called_with(
            collection_name="test_collection", vectors_config=expected_config
        )

    def test_insert(self):
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"key": "value1"}, {"key": "value2"}]
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        self.qdrant.insert(vectors=vectors, payloads=payloads, ids=ids)

        self.client_mock.upsert.assert_called_once()
        points = self.client_mock.upsert.call_args[1]["points"]

        self.assertEqual(len(points), 2)
        for point in points:
            self.assertIsInstance(point, PointStruct)

        self.assertEqual(points[0].payload, payloads[0])

    def test_search(self):
        vectors = [[0.1, 0.2]]
        mock_point = MagicMock(id=str(uuid.uuid4()), score=0.95, payload={"key": "value"})
        self.client_mock.query_points.return_value = MagicMock(points=[mock_point])

        results = self.qdrant.search(query="", vectors=vectors, limit=1)

        self.client_mock.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=vectors,
            query_filter=None,
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload, {"key": "value"})
        self.assertEqual(results[0].score, 0.95)

    def test_delete(self):
        vector_id = str(uuid.uuid4())
        self.qdrant.delete(vector_id=vector_id)

        self.client_mock.delete.assert_called_once_with(
            collection_name="test_collection",
            points_selector=PointIdsList(points=[vector_id]),
        )

    def test_update(self):
        vector_id = str(uuid.uuid4())
        updated_vector = [0.2, 0.3]
        updated_payload = {"key": "updated_value"}

        self.qdrant.update(vector_id=vector_id, vector=updated_vector, payload=updated_payload)

        self.client_mock.upsert.assert_called_once()
        point = self.client_mock.upsert.call_args[1]["points"][0]
        self.assertEqual(point.id, vector_id)
        self.assertEqual(point.vector, updated_vector)
        self.assertEqual(point.payload, updated_payload)

    def test_get(self):
        vector_id = str(uuid.uuid4())
        self.client_mock.retrieve.return_value = [{"id": vector_id, "payload": {"key": "value"}}]

        result = self.qdrant.get(vector_id=vector_id)

        self.client_mock.retrieve.assert_called_once_with(
            collection_name="test_collection", ids=[vector_id], with_payload=True
        )
        self.assertEqual(result["id"], vector_id)
        self.assertEqual(result["payload"], {"key": "value"})

    def test_list_cols(self):
        self.client_mock.get_collections.return_value = MagicMock(collections=[{"name": "test_collection"}])
        result = self.qdrant.list_cols()
        self.assertEqual(result.collections[0]["name"], "test_collection")

    def test_delete_col(self):
        self.qdrant.delete_col()
        self.client_mock.delete_collection.assert_called_once_with(collection_name="test_collection")

    def test_col_info(self):
        self.qdrant.col_info()
        self.client_mock.get_collection.assert_called_once_with(collection_name="test_collection")

    def test_create_filter_with_dict(self):
        # Test simple key-value filter
        filters = {"field": "value"}
        result = self.qdrant._create_filter(filters)
        self.assertIsInstance(result, Filter)
        self.assertEqual(len(result.must), 1)
        self.assertIsInstance(result.must[0], FieldCondition)
        self.assertEqual(result.must[0].key, "field")
        self.assertIsInstance(result.must[0].match, MatchValue)
        self.assertEqual(result.must[0].match.value, "value")

        # Test range filter
        filters = {"field": {"gte": 1, "lte": 2}}
        result = self.qdrant._create_filter(filters)
        self.assertIsInstance(result, Filter)
        self.assertEqual(len(result.must), 1)
        self.assertIsInstance(result.must[0], FieldCondition)
        self.assertEqual(result.must[0].key, "field")
        self.assertIsInstance(result.must[0].range, Range)
        self.assertEqual(result.must[0].range.gte, 1)
        self.assertEqual(result.must[0].range.lte, 2)

        # Test multiple conditions
        filters = {
            "field1": "value1",
            "field2": {"gt": 1, "lt": 2}
        }
        result = self.qdrant._create_filter(filters)
        self.assertIsInstance(result, Filter)
        self.assertEqual(len(result.must), 2)

    def test_create_filter_with_filter_base(self):
        # Test with QdrantFilter
        filter_obj = QdrantFilter().match("field", "value")
        result = self.qdrant._create_filter(filter_obj)
        self.assertIsInstance(result, Filter)
        self.assertEqual(len(result.must), 1)
        self.assertIsInstance(result.must[0], FieldCondition)
        self.assertEqual(result.must[0].key, "field")
        self.assertIsInstance(result.must[0].match, MatchValue)
        self.assertEqual(result.must[0].match.value, "value")

    def test_create_filter_with_none(self):
        result = self.qdrant._create_filter(None)
        self.assertIsNone(result)

    def tearDown(self):
        del self.qdrant


class TestQdrantFilter(unittest.TestCase):
    def setUp(self):
        self.filter = QdrantFilter()

    def test_match(self):
        filter_obj = self.filter.match("field", "value")
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].match, MatchValue)
        self.assertEqual(built_filter.must[0].match.value, "value")

    def test_match_any(self):
        filter_obj = self.filter.match_any("field", ["value1", "value2"])
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].match, MatchAny)
        self.assertEqual(built_filter.must[0].match.any, ["value1", "value2"])

    def test_match_except(self):
        filter_obj = self.filter.match_except("field", ["value1", "value2"])
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].match, MatchExcept)
        self.assertEqual(built_filter.must[0].match.except_, ["value1", "value2"])

    def test_range(self):
        filter_obj = self.filter.range("field", gt=1.0, lt=2.0)
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].range, Range)
        self.assertEqual(built_filter.must[0].range.gt, 1.0)
        self.assertEqual(built_filter.must[0].range.lt, 2.0)

    def test_has_id(self):
        filter_obj = self.filter.has_id([1, 2, 3])
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], HasIdCondition)
        self.assertEqual(built_filter.must[0].has_id, [1, 2, 3])

    def test_has_vector(self):
        filter_obj = self.filter.has_vector("vector_name")
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], HasVectorCondition)
        self.assertEqual(built_filter.must[0].has_vector, "vector_name")

    def test_is_empty(self):
        filter_obj = self.filter.is_empty("field")
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], IsEmptyCondition)
        self.assertEqual(built_filter.must[0].is_empty.key, "field")

    def test_is_null(self):
        filter_obj = self.filter.is_null("field")
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], IsNullCondition)
        self.assertEqual(built_filter.must[0].is_null.key, "field")

    def test_values_count(self):
        filter_obj = self.filter.values_count("field", gt=1, lt=5)
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].values_count, ValuesCount)
        self.assertEqual(built_filter.must[0].values_count.gt, 1)
        self.assertEqual(built_filter.must[0].values_count.lt, 5)

    def test_geo_bounding_box(self):
        filter_obj = self.filter.geo_bounding_box(
            "field",
            bottom_right={"lat": 1.0, "lon": 2.0},
            top_left={"lat": 3.0, "lon": 4.0}
        )
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].geo_bounding_box, GeoBoundingBox)
        self.assertEqual(built_filter.must[0].geo_bounding_box.bottom_right.lat, 1.0)
        self.assertEqual(built_filter.must[0].geo_bounding_box.bottom_right.lon, 2.0)
        self.assertEqual(built_filter.must[0].geo_bounding_box.top_left.lat, 3.0)
        self.assertEqual(built_filter.must[0].geo_bounding_box.top_left.lon, 4.0)

    def test_geo_radius(self):
        filter_obj = self.filter.geo_radius(
            "field",
            center={"lat": 1.0, "lon": 2.0},
            radius=1000.0
        )
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].geo_radius, GeoRadius)
        self.assertEqual(built_filter.must[0].geo_radius.center.lat, 1.0)
        self.assertEqual(built_filter.must[0].geo_radius.center.lon, 2.0)
        self.assertEqual(built_filter.must[0].geo_radius.radius, 1000.0)

    def test_geo_polygon(self):
        filter_obj = self.filter.geo_polygon(
            "field",
            exterior=[{"lat": 1.0, "lon": 2.0}, {"lat": 3.0, "lon": 4.0}],
            interiors=[[{"lat": 5.0, "lon": 6.0}, {"lat": 7.0, "lon": 8.0}]]
        )
        self.assertIsInstance(filter_obj, QdrantFilter)
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertIsInstance(built_filter.must[0], FieldCondition)
        self.assertEqual(built_filter.must[0].key, "field")
        self.assertIsInstance(built_filter.must[0].geo_polygon, GeoPolygon)
        self.assertEqual(len(built_filter.must[0].geo_polygon.exterior.points), 2)
        self.assertEqual(len(built_filter.must[0].geo_polygon.interiors), 1)
        self.assertEqual(len(built_filter.must[0].geo_polygon.interiors[0].points), 2)

    def test_must_should_must_not(self):
        filter_obj = self.filter.match("field1", "value1").must(
            self.filter.match("field2", "value2")
        ).should(
            self.filter.match("field3", "value3")
        ).must_not(
            self.filter.match("field4", "value4")
        )
        self.assertIsInstance(filter_obj, QdrantFilter)
        # This should raise an error because we're mixing direct conditions with must/should/must_not clauses
        with self.assertRaises(ValueError):
            filter_obj.build()

    def test_nested_filter(self):
        filter_obj = (self.filter.match("field1", "value1")
                      .must(self.filter.match("field2", "value2")
                            .must(self.filter.match("field3", "value3")
                                  .should(self.filter.match("field4", "value4")
                                          .must_not(self.filter.match("field5", "value5")))))
        )
        filter_obj.build()
        built_filter = filter_obj.build()
        self.assertEqual(len(built_filter.must), 1)
        self.assertEqual(len(built_filter.must[0].must), 1)
        self.assertEqual(len(built_filter.must[0].must[0].must), 1)
        self.assertEqual(len(built_filter.must[0].must[0].must[0].should), 1)
        self.assertEqual(len(built_filter.must[0].must[0].must[0].should[0].must_not), 1)

    def test_mixed_conditions_error(self):
        filter_obj = self.filter.match("field1", "value1")
        filter_obj._must.append(FieldCondition(key="field2", match=MatchValue(value="value2")))
        with self.assertRaises(ValueError):
            filter_obj.build()

    def tearDown(self):
        del self.filter
