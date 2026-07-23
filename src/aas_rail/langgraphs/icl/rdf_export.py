from urllib.parse import quote, urlsplit, urlunsplit

from basyx.aas import model
from rdflib import RDF, URIRef



URI_SAFE_CHARS = ":/?#[]@!$&'()*+,;=-._~%"


# def _load_notebook_rdf_serializer():
#     for parent in Path(__file__).resolve().parents:
#         candidate = parent / "notebooks" / "rdf_serialization.py"
#         if candidate.exists():
#             spec = importlib.util.spec_from_file_location("vamos_notebook_rdf_serialization", candidate)
#             module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(module)
#             return module
#     raise ImportError("Could not find notebooks/rdf_serialization.py")


# _rdf_serialization = _load_notebook_rdf_serializer()

from .rdf_serialization import AASToRDFEncoder

def safe_uri_ref(value) -> URIRef:
    text = str(value or "").strip()
    parsed = urlsplit(text)
    safe = URI_SAFE_CHARS.replace("#", "")
    path = quote(parsed.path, safe=safe)
    query = quote(parsed.query, safe=safe)
    fragment = quote(parsed.fragment, safe=safe)
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, query, fragment))
    return URIRef(normalized)


class SafeAASToRDFEncoder(AASToRDFEncoder):
    def _concept_description_to_rdf(self, obj: model.ConceptDescription) -> None:
        subject = safe_uri_ref(obj.id)
        self.graph.add((subject, RDF.type, self.aas["ConceptDescription"]))
        self._abstract_classes_to_rdf(obj, subject)
        if obj.is_case_of:
            for reference in obj.is_case_of:
                self._reference_to_rdf(reference, subject, self.aas["ConceptDescription/isCaseOf"])

    def _asset_administration_shell_to_rdf(self, obj: model.AssetAdministrationShell) -> None:
        subject = safe_uri_ref(obj.id)
        self.graph.add((subject, RDF.type, self.aas["AssetAdministrationShell"]))
        self._abstract_classes_to_rdf(obj, subject)
        if obj.derived_from:
            self._reference_to_rdf(obj.derived_from, subject, self.aas["AssetAdministrationShell/derivedFrom"])
        if obj.asset_information:
            self._asset_information_to_rdf(obj.asset_information, subject)
        if obj.submodel:
            for reference in obj.submodel:
                self._reference_to_rdf(reference, subject, self.aas["AssetAdministrationShell/submodels"])

    def _submodel_to_rdf(self, obj: model.Submodel) -> None:
        subject = safe_uri_ref(obj.id)
        self.graph.add((subject, RDF.type, self.aas["Submodel"]))
        self._abstract_classes_to_rdf(obj, subject)
        if obj.submodel_element:
            for submodel_element in obj.submodel_element:
                self._submodel_element_to_rdf(submodel_element, subject, self.aas["Submodel/submodelElements"])

    def _range_to_rdf(self, obj: model.Range, parent) -> None:
        self.graph.add((parent, RDF.type, self.aas["Range"]))
        self._abstract_classes_to_rdf(obj, parent)
        self.graph.add((
            parent,
            self.aas["Range/valueType"],
            self.aas[f"DataTypeDefXsd/{model.datatypes.XSD_TYPE_NAMES[obj.value_type]}"],
        ))
        if obj.min is not None:
            self._value_to_rdf(obj.min, obj.value_type, parent, self.aas["Range/min"])
        if obj.max is not None:
            self._value_to_rdf(obj.max, obj.value_type, parent, self.aas["Range/max"])


def object_store_to_turtle(object_store: model.AbstractObjectStore) -> str:
    encoder = SafeAASToRDFEncoder()
    encoder.object_store_to_rdflib_graph(object_store)
    return encoder.graph.serialize(format="turtle")


def object_store_to_turtle_bytes(object_store: model.AbstractObjectStore) -> bytes:
    return object_store_to_turtle(object_store).encode("utf-8")

