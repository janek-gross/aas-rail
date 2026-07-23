"""
Cypher queries for retrieving ICL property examples from the n10s RDF graph.
"""

from __future__ import annotations

from typing import Any, Iterable


def _cypher_string_values(value_expr: str) -> str:
    return (
        "CASE "
        f"WHEN {value_expr} IS NULL THEN [] "
        f"WHEN {value_expr} IS :: LIST<ANY> THEN toStringList({value_expr}) "
        f"ELSE [toString({value_expr})] END"
    )


def _cypher_property_string_values(node_expr: str, key_expr: str = "key") -> str:
    return _cypher_string_values(f"{node_expr}[{key_expr}]")


def _instruction_values(suffix: str) -> str:
    return (
        f'reduce(values = [], key IN [key IN keys(instruction) WHERE key ENDS WITH "__{suffix}"] | '
        f"values + {_cypher_property_string_values('instruction')})"
    )


def _instruction_scalar_value(suffix: str) -> str:
    return f"head({_instruction_values(suffix)})"


def _instruction_value_filter(suffix: str, param_name: str, operator: str) -> str:
    return (
        f'any(key IN keys(instruction) WHERE key ENDS WITH "__{suffix}" '
        f"AND any(value IN {_cypher_property_string_values('instruction')} "
        f"WHERE value IS NOT NULL AND value {operator} ${param_name}))"
    )


def _build_extraction_instruction_projection() -> str:
    fields = [
        ("groupKey", "groupKey", "scalar"),
        ("responseFieldName", "responseFieldName", "scalar"),
        ("semanticId", "semanticId", "scalar"),
        ("idShort", "idShort", "scalar"),
        ("unit", "unit", "scalar"),
        ("propertyRecordId", "propertyRecordId", "scalar"),
        ("propertyRecordCount", "propertyRecordCount", "scalar"),
        ("propertyName", "propertyName", "scalar"),
        ("propertyPaths", "propertyPath", "list"),
        ("grounding", "grounding", "scalar"),
        ("evidence", "evidence", "scalar"),
        ("extractionRule", "extractionRule", "scalar"),
        ("formattingRule", "formattingRule", "scalar"),
        ("avoid", "avoid", "list"),
        ("helperGenerationHash", "helperGenerationHash", "scalar"),
        ("helperGenerationConfigJson", "helperGenerationConfigJson", "scalar"),
        ("helperProvider", "helperProvider", "scalar"),
        ("helperModel", "helperModel", "scalar"),
        ("helperArtifactIds", "helperArtifactId", "list"),
    ]
    lines = ["    RETURN [instruction IN instructionNodes | {"]
    for index, (field_name, suffix, cardinality) in enumerate(fields):
        value_expr = (
            _instruction_values(suffix)
            if cardinality == "list"
            else _instruction_scalar_value(suffix)
        )
        comma = "," if index < len(fields) - 1 else ""
        lines.append(f"        {field_name}: {value_expr}{comma}")
    lines.append("    }] AS extractionInstructions")
    return "\n".join(lines)


_PROPERTY_VALUES_WITH_METADATA_CYPHER_TEMPLATE = """
WITH $properties AS requestedProperties,
     coalesce($target_eclass_id, "") AS targetEclassId,
     coalesce($target_manufacturer_name, "") AS targetManufacturerName
MATCH (element)
WHERE any(label IN labels(element) WHERE label ENDS WITH "__Property" OR label ENDS WITH "__MultiLanguageProperty" OR label ENDS WITH "__Range")
WITH element, requestedProperties, targetEclassId, targetManufacturerName,
     labels(element) AS elementLabels,
     [key IN keys(element) WHERE key ENDS WITH "__idShort" | toString(element[key])] AS idShorts,
     [key IN keys(element) WHERE key ENDS WITH "__value" | toString(element[key])] AS directValues,
     [key IN keys(element) WHERE key ENDS WITH "__min" | toString(element[key])] AS minValues,
     [key IN keys(element) WHERE key ENDS WITH "__max" | toString(element[key])] AS maxValues,
     coalesce(element.iclSourceNames, []) AS sourceNames
OPTIONAL MATCH (element)-[semanticRel]->()-[keyRel]->(semanticKey)
WHERE type(semanticRel) ENDS WITH "__semanticId" AND type(keyRel) ENDS WITH "__keys"
WITH element, requestedProperties, targetEclassId, targetManufacturerName, elementLabels, idShorts,
     directValues, minValues, maxValues, sourceNames,
     collect(DISTINCT semanticKey) AS semanticKeyNodes
WITH element, requestedProperties, targetEclassId, targetManufacturerName, elementLabels, idShorts,
     directValues, minValues, maxValues, sourceNames,
     reduce(values = [], node IN semanticKeyNodes |
         values + [key IN keys(node) WHERE key ENDS WITH "__value" | toString(node[key])]
     ) AS semanticIds
OPTIONAL MATCH (element)-[valueRel]->(langValue)
WHERE type(valueRel) ENDS WITH "__value"
  AND any(label IN labels(langValue) WHERE label ENDS WITH "__LangStringTextType" OR label ENDS WITH "__LangStringNameType")
WITH element, requestedProperties, targetEclassId, targetManufacturerName, elementLabels, idShorts,
     directValues, minValues, maxValues, sourceNames, semanticIds,
     collect(DISTINCT {
         text: head([key IN keys(langValue) WHERE key ENDS WITH "__text" | toString(langValue[key])]),
         language: head([key IN keys(langValue) WHERE key ENDS WITH "__language" | toString(langValue[key])])
     }) AS langValues
UNWIND requestedProperties AS requested
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested,
     [value IN directValues WHERE value IS NOT NULL AND trim(value) <> ""] AS cleanDirectValues,
     [value IN minValues WHERE value IS NOT NULL AND trim(value) <> "" | "min: " + value] AS cleanMinValues,
     [value IN maxValues WHERE value IS NOT NULL AND trim(value) <> "" | "max: " + value] AS cleanMaxValues,
     [value IN langValues WHERE value.text IS NOT NULL AND trim(value.text) <> ""] AS cleanLangValues
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested, cleanLangValues,
     cleanDirectValues + cleanMinValues + cleanMaxValues + [value IN cleanLangValues | value.text] AS allValues,
     toLower(coalesce(requested.property_id, "")) AS requestedId,
     toLower(coalesce(requested.property_name, "")) AS requestedName
WHERE size(allValues) > 0
  AND (
      requestedId <> ""
      AND (
          any(semanticId IN semanticIds WHERE toLower(semanticId) = requestedId OR toLower(semanticId) ENDS WITH requestedId)
          OR any(idShort IN idShorts WHERE toLower(idShort) = requestedId)
      )
      OR requestedName <> ""
      AND any(idShort IN idShorts WHERE toLower(idShort) = requestedName)
  )
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested, cleanLangValues, allValues,
     CASE
         WHEN any(semanticId IN semanticIds WHERE toLower(semanticId) = requestedId OR toLower(semanticId) ENDS WITH requestedId) THEN "semantic_id"
         WHEN any(idShort IN idShorts WHERE toLower(idShort) = requestedId) THEN "id_short"
         ELSE "name"
     END AS matchedBy
OPTIONAL MATCH submodelPath = (submodel)-[*1..16]->(element)
WHERE any(label IN labels(submodel) WHERE label ENDS WITH "__Submodel")
  AND all(rel IN relationships(submodelPath)
      WHERE type(rel) ENDS WITH "__submodelElements"
         OR type(rel) ENDS WITH "__value"
         OR type(rel) ENDS WITH "__statements"
  )
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested, cleanLangValues, allValues, matchedBy, submodel,
     CASE WHEN submodelPath IS NULL THEN 999 ELSE length(submodelPath) END AS submodelDepth,
     CASE
         WHEN submodelPath IS NULL THEN []
         ELSE [node IN nodes(submodelPath) |
             head([key IN keys(node) WHERE key ENDS WITH "__idShort" | toString(node[key])])
         ]
     END AS pathIdShorts
ORDER BY submodelDepth ASC
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested, cleanLangValues, allValues, matchedBy,
     head([
         info IN collect(DISTINCT {
             uri: submodel.uri,
             idShort: CASE
                 WHEN submodel IS NULL THEN null
                 ELSE head([key IN keys(submodel) WHERE key ENDS WITH "__idShort" | toString(submodel[key])])
             END,
             path: [pathIdShort IN pathIdShorts WHERE pathIdShort IS NOT NULL AND trim(pathIdShort) <> ""]
         })
         WHERE info.uri IS NOT NULL OR info.idShort IS NOT NULL OR size(info.path) > 0
     ]) AS submodelInfo
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds, sourceNames,
     requested, cleanLangValues, allValues, matchedBy, submodelInfo,
     [candidate IN coalesce(submodelInfo.path, [])
         + [coalesce(submodelInfo.idShort, ""), coalesce(submodelInfo.uri, "")]
      | replace(replace(replace(toLower(toString(candidate)), " ", ""), "_", ""), "-", "")
     ] AS pathIdentifiers
WHERE (
      NOT coalesce($technical_data_only, false)
      OR any(identifier IN pathIdentifiers WHERE identifier CONTAINS "technicaldata")
  )
  AND (
      NOT coalesce($technical_properties_only, false)
      OR any(identifier IN pathIdentifiers WHERE identifier CONTAINS "technicalproperties")
  )
UNWIND CASE WHEN size(sourceNames) = 0 THEN [""] ELSE sourceNames END AS productSource
OPTIONAL MATCH (meta)
WHERE productSource <> ""
  AND productSource IN coalesce(meta.iclSourceNames, [])
  AND any(label IN labels(meta) WHERE label ENDS WITH "__Property" OR label ENDS WITH "__MultiLanguageProperty")
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo, meta,
     [key IN keys(meta) WHERE key ENDS WITH "__idShort" | toString(meta[key])] AS metaIdShorts,
     [key IN keys(meta) WHERE key ENDS WITH "__value" | toString(meta[key])] AS metaDirectValues
OPTIONAL MATCH (meta)-[metaValueRel]->(metaLangValue)
WHERE type(metaValueRel) ENDS WITH "__value"
  AND any(label IN labels(metaLangValue) WHERE label ENDS WITH "__LangStringTextType" OR label ENDS WITH "__LangStringNameType")
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo, metaIdShorts, metaDirectValues,
     collect(DISTINCT head([key IN keys(metaLangValue) WHERE key ENDS WITH "__text" | toString(metaLangValue[key])])) AS metaLangTexts
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo,
     collect(DISTINCT {
         idShorts: metaIdShorts,
         values: [value IN metaDirectValues + metaLangTexts WHERE value IS NOT NULL AND trim(value) <> ""]
     }) AS metadataEntries
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "ProductClassId")
             THEN entry.values ELSE []
         END
     ) AS productClassIdValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "ProductGroup")
             THEN entry.values ELSE []
         END
     ) AS productGroupValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "ClassId")
             THEN entry.values ELSE []
         END
     ) AS classIdValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "ManufacturerName")
             THEN entry.values ELSE []
         END
     ) AS manufacturerNameValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "Company")
             THEN entry.values ELSE []
         END
     ) AS companyValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "NameOfSupplier")
             THEN entry.values ELSE []
         END
     ) AS supplierValues,
     reduce(values = [], entry IN metadataEntries |
         values + CASE
             WHEN any(idShort IN entry.idShorts WHERE idShort = "Brand")
             THEN entry.values ELSE []
         END
     ) AS brandValues
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo,
     [value IN productClassIdValues + productGroupValues + classIdValues
         WHERE value IS NOT NULL AND trim(value) <> ""
           AND (value =~ ".*[0-9]{8}.*" OR value =~ ".*[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}.*")
     ] AS eclassCandidates,
     [value IN productClassIdValues + productGroupValues + classIdValues WHERE value IS NOT NULL AND trim(value) <> ""] AS classIdCandidates,
     [value IN manufacturerNameValues + companyValues + supplierValues + brandValues WHERE value IS NOT NULL AND trim(value) <> ""] AS cleanManufacturerNames
WITH element, targetEclassId, targetManufacturerName, elementLabels, idShorts, semanticIds,
     productSource, requested, cleanLangValues, allValues, matchedBy, submodelInfo,
     coalesce(head(eclassCandidates), head(classIdCandidates)) AS eclassId,
     head(cleanManufacturerNames) AS manufacturerName
WITH *,
     CASE
         WHEN targetEclassId <> "" AND eclassId IS NOT NULL
              AND replace(toLower(eclassId), "-", "") = replace(toLower(targetEclassId), "-", "") THEN 2
         ELSE 0
     END
     +
     CASE
         WHEN targetManufacturerName <> "" AND manufacturerName IS NOT NULL
              AND toLower(manufacturerName) CONTAINS toLower(targetManufacturerName) THEN 1
         ELSE 0
     END AS similarityScore
CALL {
    WITH element
    OPTIONAL MATCH (element)-[instructionRel]->(instruction)
    WHERE type(instructionRel) ENDS WITH "__hasExtractionInstruction"
      AND (
          coalesce($helper_generation_hash, "") = ""
          OR __HELPER_GENERATION_HASH_FILTER__
      )
      AND (
          coalesce($helper_artifact_id, "") = ""
          OR __HELPER_ARTIFACT_ID_FILTER__
      )
      AND (
          coalesce($helper_provider, "") = ""
          OR __HELPER_PROVIDER_FILTER__
      )
      AND (
          coalesce($helper_model, "") = ""
          OR __HELPER_MODEL_FILTER__
      )
    WITH [node IN collect(DISTINCT instruction) WHERE node IS NOT NULL] AS instructionNodes
__EXTRACTION_INSTRUCTION_PROJECTION__
}
WITH requested, matchedBy, productSource, element, idShorts, semanticIds, submodelInfo,
     allValues, cleanLangValues, eclassId, manufacturerName, similarityScore, extractionInstructions
WHERE (
      coalesce($helper_generation_hash, "") = ""
      AND coalesce($helper_artifact_id, "") = ""
      AND coalesce($helper_provider, "") = ""
      AND coalesce($helper_model, "") = ""
  )
  OR size(extractionInstructions) > 0
RETURN
    requested.property_id AS requestedPropertyId,
    requested.property_name AS requestedPropertyName,
    matchedBy,
    productSource AS sourceName,
    element.uri AS elementUri,
    head(idShorts) AS propertyIdShort,
    head(semanticIds) AS semanticId,
    submodelInfo.idShort AS submodelIdShort,
    submodelInfo.uri AS submodelUri,
    submodelInfo.path AS elementPath,
    allValues AS values,
    cleanLangValues AS localizedValues,
    eclassId,
    manufacturerName,
    similarityScore,
    extractionInstructions
ORDER BY similarityScore DESC, sourceName, requestedPropertyName, propertyIdShort
LIMIT coalesce($limit, 500)
"""


PROPERTY_VALUES_WITH_METADATA_CYPHER = (
    _PROPERTY_VALUES_WITH_METADATA_CYPHER_TEMPLATE
    .replace(
        "__HELPER_GENERATION_HASH_FILTER__",
        _instruction_value_filter("helperGenerationHash", "helper_generation_hash", "="),
    )
    .replace(
        "__HELPER_ARTIFACT_ID_FILTER__",
        _instruction_value_filter("helperArtifactId", "helper_artifact_id", "CONTAINS"),
    )
    .replace(
        "__HELPER_PROVIDER_FILTER__",
        _instruction_value_filter("helperProvider", "helper_provider", "="),
    )
    .replace(
        "__HELPER_MODEL_FILTER__",
        _instruction_value_filter("helperModel", "helper_model", "="),
    )
    .replace("__EXTRACTION_INSTRUCTION_PROJECTION__", _build_extraction_instruction_projection())
)


PRODUCT_METADATA_CYPHER = """
MATCH (resource)
WHERE size(coalesce(resource.iclSourceNames, [])) > 0
UNWIND resource.iclSourceNames AS sourceName
WITH DISTINCT sourceName
MATCH (meta)
WHERE sourceName IN coalesce(meta.iclSourceNames, [])
  AND any(label IN labels(meta) WHERE label ENDS WITH "__Property" OR label ENDS WITH "__MultiLanguageProperty")
WITH sourceName, meta,
     [key IN keys(meta) WHERE key ENDS WITH "__idShort" | toString(meta[key])] AS idShorts,
     [key IN keys(meta) WHERE key ENDS WITH "__value" | toString(meta[key])] AS directValues
OPTIONAL MATCH (meta)-[valueRel]->(langValue)
WHERE type(valueRel) ENDS WITH "__value"
  AND any(label IN labels(langValue) WHERE label ENDS WITH "__LangStringTextType" OR label ENDS WITH "__LangStringNameType")
WITH sourceName, idShorts, directValues,
     collect(DISTINCT head([key IN keys(langValue) WHERE key ENDS WITH "__text" | toString(langValue[key])])) AS langTexts
WITH sourceName,
     collect(DISTINCT {
         idShorts: idShorts,
         values: [value IN directValues + langTexts WHERE value IS NOT NULL AND trim(value) <> ""]
     }) AS entries
RETURN
    sourceName,
    coalesce(
        head([value IN reduce(values = [], entry IN entries |
            values + CASE
                WHEN any(idShort IN entry.idShorts WHERE idShort = "ProductClassId")
                THEN entry.values ELSE []
            END
        ) + reduce(values = [], entry IN entries |
            values + CASE
                WHEN any(idShort IN entry.idShorts WHERE idShort = "ProductGroup")
                THEN entry.values ELSE []
            END
        ) + reduce(values = [], entry IN entries |
            values + CASE
                WHEN any(idShort IN entry.idShorts WHERE idShort = "ClassId")
                THEN entry.values ELSE []
            END
        )
        WHERE value IS NOT NULL AND trim(value) <> ""
          AND (value =~ ".*[0-9]{8}.*" OR value =~ ".*[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}.*")
        ]),
        head([value IN reduce(values = [], entry IN entries |
            values + CASE
                WHEN any(idShort IN entry.idShorts WHERE idShort IN ["ProductClassId", "ProductGroup", "ClassId"])
                THEN entry.values ELSE []
            END
        ) WHERE value IS NOT NULL AND trim(value) <> ""])
    ) AS eclassId,
    head(reduce(values = [], entry IN entries |
        values + CASE
            WHEN any(idShort IN entry.idShorts WHERE idShort = "ManufacturerName")
            THEN entry.values ELSE []
        END
    ) + reduce(values = [], entry IN entries |
        values + CASE
            WHEN any(idShort IN entry.idShorts WHERE idShort = "Company")
            THEN entry.values ELSE []
        END
    ) + reduce(values = [], entry IN entries |
        values + CASE
            WHEN any(idShort IN entry.idShorts WHERE idShort = "NameOfSupplier")
            THEN entry.values ELSE []
        END
    ) + reduce(values = [], entry IN entries |
        values + CASE
            WHEN any(idShort IN entry.idShorts WHERE idShort = "Brand")
            THEN entry.values ELSE []
        END
    )) AS manufacturerName
ORDER BY sourceName
"""


DATASHEET_EMBEDDINGS_CYPHER = """
MATCH (embedding)
WHERE any(label IN labels(embedding) WHERE label ENDS WITH "__ProductDatasheetEmbedding")
WITH embedding,
     head([key IN keys(embedding) WHERE key ENDS WITH "__sourceName" | toString(embedding[key])]) AS sourceName,
     head([key IN keys(embedding) WHERE key ENDS WITH "__pdfName" | toString(embedding[key])]) AS pdfName,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingProvider" | toString(embedding[key])]) AS embeddingProvider,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingModel" | toString(embedding[key])]) AS embeddingModel,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingConfigHash" | toString(embedding[key])]) AS embeddingConfigHash,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingConfigJson" | toString(embedding[key])]) AS embeddingConfigJson,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingJson" | toString(embedding[key])]) AS embeddingJson,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingDimensions" | toString(embedding[key])]) AS embeddingDimensions,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingChunkSize" | toString(embedding[key])]) AS embeddingChunkSize,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingChunkOverlap" | toString(embedding[key])]) AS embeddingChunkOverlap,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingMaxPdfChars" | toString(embedding[key])]) AS embeddingMaxPdfChars,
     head([key IN keys(embedding) WHERE key ENDS WITH "__embeddingChunkCount" | toString(embedding[key])]) AS embeddingChunkCount,
     head([key IN keys(embedding) WHERE key ENDS WITH "__textCharCount" | toString(embedding[key])]) AS textCharCount
WHERE sourceName IS NOT NULL
  AND trim(sourceName) <> ""
  AND (size($source_names) = 0 OR sourceName IN $source_names)
  AND (
      coalesce($embedding_config_hash, "") = ""
      OR embeddingConfigHash = $embedding_config_hash
  )
RETURN
    sourceName,
    pdfName,
    embeddingProvider,
    embeddingModel,
    embeddingConfigHash,
    embeddingConfigJson,
    embeddingJson,
    toInteger(embeddingDimensions) AS embeddingDimensions,
    toInteger(embeddingChunkSize) AS embeddingChunkSize,
    toInteger(embeddingChunkOverlap) AS embeddingChunkOverlap,
    toInteger(embeddingMaxPdfChars) AS embeddingMaxPdfChars,
    toInteger(embeddingChunkCount) AS embeddingChunkCount,
    toInteger(textCharCount) AS textCharCount
ORDER BY sourceName
"""


def normalize_property_definition_params(definitions: Any) -> list[dict[str, str]]:
    """Convert supported uploaded definition shapes to Cypher query parameters."""
    return [_normalize_property_definition_item(item) for item in _iter_property_definition_items(definitions)]


def query_property_values_with_metadata(
    definitions: Any,
    uri: str,
    user: str,
    password: str,
    target_eclass_id: str = "",
    target_manufacturer_name: str = "",
    limit: int = 500,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
    helper_generation_hash: str = "",
    helper_artifact_id: str = "",
    helper_provider: str = "",
    helper_model: str = "",
) -> list[dict[str, Any]]:
    """Execute the ICL property-value query for uploaded property definitions."""
    from neo4j import GraphDatabase

    from .neo4j_connection import connect_neo4j

    properties = normalize_property_definition_params(definitions)
    driver = None
    try:
        driver, _, _ = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            records = session.run(
                PROPERTY_VALUES_WITH_METADATA_CYPHER,
                properties=properties,
                target_eclass_id=target_eclass_id,
                target_manufacturer_name=target_manufacturer_name,
                limit=limit,
                technical_data_only=technical_data_only,
                technical_properties_only=technical_properties_only,
                helper_generation_hash=helper_generation_hash,
                helper_artifact_id=helper_artifact_id,
                helper_provider=helper_provider,
                helper_model=helper_model,
            )
            return [record.data() for record in records]
    finally:
        if driver is not None:
            driver.close()


def query_product_metadata(uri: str, user: str, password: str) -> list[dict[str, Any]]:
    """Return product-level metadata detected in the ICL graph."""
    from neo4j import GraphDatabase

    from .neo4j_connection import connect_neo4j

    driver = None
    try:
        driver, _, _ = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            return [record.data() for record in session.run(PRODUCT_METADATA_CYPHER)]
    finally:
        if driver is not None:
            driver.close()


def query_datasheet_embeddings(
    uri: str,
    user: str,
    password: str,
    embedding_config_hash: str = "",
    source_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return source-level datasheet embeddings stored in the ICL graph."""
    from neo4j import GraphDatabase

    from .neo4j_connection import connect_neo4j

    driver = None
    try:
        driver, _, _ = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            return [
                record.data()
                for record in session.run(
                    DATASHEET_EMBEDDINGS_CYPHER,
                    embedding_config_hash=embedding_config_hash,
                    source_names=list(source_names or []),
                )
            ]
    finally:
        if driver is not None:
            driver.close()


def _iter_property_definition_items(definitions: Any) -> Iterable[Any]:
    if isinstance(definitions, list):
        return definitions

    if not isinstance(definitions, dict):
        return []

    if isinstance(definitions.get("properties"), dict):
        return definitions["properties"].values()
    if isinstance(definitions.get("properties"), list):
        return definitions["properties"]
    if isinstance(definitions.get("definitions"), dict):
        return definitions["definitions"].values()
    if isinstance(definitions.get("definitions"), list):
        return definitions["definitions"]
    if isinstance(definitions.get("property_definitions"), dict):
        return definitions["property_definitions"].values()
    if isinstance(definitions.get("property_definitions"), list):
        return definitions["property_definitions"]
    if isinstance(definitions.get("classes"), dict):
        properties = definitions.get("properties", {})
        seen = set()
        items = []
        for class_item in definitions["classes"].values():
            for property_ref in class_item.get("properties", []) if isinstance(class_item, dict) else []:
                property_id = str(property_ref)
                if property_id in seen:
                    continue
                seen.add(property_id)
                items.append(properties.get(property_id, {"id": property_id}) if isinstance(properties, dict) else {"id": property_id})
        return items

    if all(isinstance(value, dict) for value in definitions.values()):
        return definitions.values()
    return []


def _normalize_property_definition_item(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        return {"property_id": str(item), "property_name": "", "property_definition": ""}

    property_id = str(item.get("id") or item.get("property_id") or item.get("semantic_id") or "")
    property_name = _first_text(item.get("name"))
    property_definition = _first_text(item.get("definition"))
    return {
        "property_id": property_id,
        "property_name": property_name,
        "property_definition": property_definition,
    }


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if value.get("en") is not None:
            return str(value["en"])
        for item in value.values():
            if item is not None:
                return str(item)
        return ""
    return str(value)
