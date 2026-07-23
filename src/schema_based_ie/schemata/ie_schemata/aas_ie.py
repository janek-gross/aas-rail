from typing import Union, Type, Literal, Optional
from pydantic import BaseModel, RootModel, Field, create_model, ConfigDict, model_validator
from enum import Enum


# ---------- Technical Properties ----------

class PropertyItem(BaseModel):
    # property: str
    # value: Union[str, int, float, list[Union[str, int, float]]]
    reference: Optional[str] = None
    unit: Optional[str] = None
    value: Union[str, int, float]

class PropertyEntry(BaseModel):
    name: str
    item: PropertyItem

class TechnicalPropertySchema(BaseModel):
    properties: list[PropertyEntry] = Field(...,
        description="Mapping of property name to property value"
    )

    @model_validator(mode="before")
    @classmethod
    def accept_dict_kwargs(cls, data):
        # Allows: TechnicalPropertySchema(**label_dict)
        if "properties" not in data:
            return {
                "properties": [
                    {"name": k, "item": v}
                    for k, v in data.items()
                ]
            }
        return data
    def as_dict(self) -> dict[str, PropertyItem]:
        return {p.name: p.item for p in self.properties}

def schema_factory(
    property_definitions: list[dict] = [],
    model_name: str = 'InstanceBasedExtractionSchema'
    ) -> Type[BaseModel]:
    fields = {}
    for prop in property_definitions:
        fields[prop['id']] = (
            PropertyItem,
            Field(
                ...,
                definition=prop['definition'].get('en')
            )
        )
    return create_model(model_name, **fields)


TYPE_MAP = {
    "": Union[str, int, float],
    "str": str,
    "string": str,
    "int": int,
    "decimal": float,
    "float": float,
    "bool": bool,
}


def singleton_enum(enum_name: str, value: str):
    return Enum(enum_name, {"VALUE": value})

def partially_completed_schema_factory(
    property_definitions: list[dict],
    model_name: str = 'PartiallyCompletedInstanceBasedExtractionSchema',
    ) -> Type[BaseModel]:
    fields = {}
    for prop in property_definitions:
        value_fields = {}
        prop_name = prop['id']
        if "definition" in prop:
            DefinitionEnum = singleton_enum(
                f"{prop_name.capitalize()}Definition",
                prop['definition']['en']
            )
            value_fields['definition'] = (
                DefinitionEnum,
                DefinitionEnum.VALUE
            )
        value_fields["reference"] = (Optional[str], None)
        # Optional unit (not frozen in this example)
        value_fields["unit"] = (Optional[str], prop.get("unit"))
        value_fields['value'] = (TYPE_MAP.get(prop.get('type', ''), Union[str, int, float]), ...)

        ValueModel = create_model(
            f"{prop_name.capitalize()}Value",
            **value_fields,
            __config__=ConfigDict(frozen=True)
        )

        fields[prop_name] = (ValueModel, ...)

    return create_model(model_name, **fields, __config__=ConfigDict(frozen=True))


AAS_SCHEMA_REGISTRY = {
    "aas_elastic": lambda **_: TechnicalPropertySchema,
    "aas_factory": schema_factory,
    "aas_partially_completed_factory": partially_completed_schema_factory
}
