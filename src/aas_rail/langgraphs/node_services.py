# TODO interfaces instead of inheritance

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from aas_rail.model_clients.llm_clients import EmbeddingClient
import chromadb


# class EmbeddingService(ABC):
#     @abstractmethod
#     def embed(self, texts: list[str]) -> list[list[float]]:
#         pass

class VectorStore(ABC):
    @abstractmethod
    def add(self, texts, embeddings, ids):
        pass
    
    @abstractmethod
    def search(self, queries, embedder: EmbeddingClient | None, top_k):
        pass



class QueryGenerator(ABC):
    @abstractmethod
    def generate(self, property_definition: str) -> list[str]:
        pass

class SchemaBasedExtractor(ABC):
    @abstractmethod
    def extract(self, retrieved_text, property_definition):
        pass



@dataclass
class IEServices:
    schema_based_extractor: SchemaBasedExtractor


@dataclass
class RAGServices(IEServices):
    embedder: EmbeddingClient
    vector_store: VectorStore
    query_generator: QueryGenerator