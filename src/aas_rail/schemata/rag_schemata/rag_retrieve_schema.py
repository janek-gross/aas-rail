from enum import Enum
from pydantic import BaseModel, Field, create_model
from typing import Type

class PropertyQueries(BaseModel):
    queries: list[str] = Field(..., description="The list of suggested queries for a single property")

class QueriesBatch(BaseModel):
    query_lists: list[PropertyQueries] = Field(..., description="The list of PropertyQueries lists corresponding to a batch of properties")

def property_queries_factory(n_queries: int) -> Type[BaseModel]:
    """
    Creates a PropertyQueries model with exactly n_queries fields:
        query_0, query_1, ..., query_{n-1}
    """
    fields = {
        f"query_{i}": (
            str,
            Field(..., description=f"Suggested query #{i} for this property"),
        )
        for i in range(n_queries)
    }

    return create_model("PropertyQueries", **fields)

def queries_batch_factory(
    batch_size: int,
    n_queries_per_property: int
    ) -> Type[BaseModel]:
    """
    Create a Pydantic model with exactly `batch_size` PropertyQueries fields.

    The generated model will have fields:
        query_list_0, query_list_1, ..., query_list_{batch_size-1}
    """
    PropertyQueriesModel = property_queries_factory(n_queries_per_property)

    fields = {
        f"query_list_property_{i}": (
            PropertyQueriesModel,
            Field(
                ...,
                description=f"Query list for property #{i}"
            ),
        )
        for i in range(batch_size)
    }

    return create_model("ComparisonBatch", **fields)



RETRIEVE_SCHEMA_REGISTRY = {
    'retrieve_elastic': lambda **_: QueriesBatch,
    'retrieve_factory': queries_batch_factory
}
