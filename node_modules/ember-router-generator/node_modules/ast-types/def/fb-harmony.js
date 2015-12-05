require("./core");
var types = require("../lib/types");
var def = types.Type.def;
var or = types.Type.or;
var builtin = types.builtInTypes;
var isString = builtin.string;
var isBoolean = builtin.boolean;
var defaults = require("../lib/shared").defaults;

def("XJSAttribute")
    .bases("Node")
    .build("name", "value")
    .field("name", or(def("XJSIdentifier"), def("XJSNamespacedName")))
    .field("value", or(
        def("Literal"), // attr="value"
        def("XJSExpressionContainer"), // attr={value}
        null // attr= or just attr
    ), defaults["null"]);

def("XJSIdentifier")
    .bases("Node")
    .build("name")
    .field("name", isString);

def("XJSNamespacedName")
    .bases("Node")
    .build("namespace", "name")
    .field("namespace", def("XJSIdentifier"))
    .field("name", def("XJSIdentifier"));

def("XJSMemberExpression")
    .bases("MemberExpression")
    .build("object", "property")
    .field("object", or(def("XJSIdentifier"), def("XJSMemberExpression")))
    .field("property", def("XJSIdentifier"))
    .field("computed", isBoolean, defaults.false);

var XJSElementName = or(
    def("XJSIdentifier"),
    def("XJSNamespacedName"),
    def("XJSMemberExpression")
);

def("XJSSpreadAttribute")
    .bases("Node")
    .build("argument")
    .field("argument", def("Expression"));

var XJSAttributes = [or(
    def("XJSAttribute"),
    def("XJSSpreadAttribute")
)];

def("XJSExpressionContainer")
    .bases("Expression")
    .build("expression")
    .field("expression", def("Expression"));

def("XJSElement")
    .bases("Expression")
    .build("openingElement", "closingElement", "children")
    .field("openingElement", def("XJSOpeningElement"))
    .field("closingElement", or(def("XJSClosingElement"), null), defaults["null"])
    .field("children", [or(
        def("XJSElement"),
        def("XJSExpressionContainer"),
        def("XJSText"),
        def("Literal") // TODO Esprima should return XJSText instead.
    )], defaults.emptyArray)
    .field("name", XJSElementName, function() {
        // Little-known fact: the `this` object inside a default function
        // is none other than the partially-built object itself, and any
        // fields initialized directly from builder function arguments
        // (like openingElement, closingElement, and children) are
        // guaranteed to be available.
        return this.openingElement.name;
    })
    .field("selfClosing", isBoolean, function() {
        return this.openingElement.selfClosing;
    })
    .field("attributes", XJSAttributes, function() {
        return this.openingElement.attributes;
    });

def("XJSOpeningElement")
    .bases("Node") // TODO Does this make sense? Can't really be an XJSElement.
    .build("name", "attributes", "selfClosing")
    .field("name", XJSElementName)
    .field("attributes", XJSAttributes, defaults.emptyArray)
    .field("selfClosing", isBoolean, defaults["false"]);

def("XJSClosingElement")
    .bases("Node") // TODO Same concern.
    .build("name")
    .field("name", XJSElementName);

def("XJSText")
    .bases("Literal")
    .build("value")
    .field("value", isString);

def("XJSEmptyExpression").bases("Expression").build();

// Type Annotations
def("Type")
  .bases("Node");

def("AnyTypeAnnotation")
  .bases("Type");

def("VoidTypeAnnotation")
  .bases("Type");

def("NumberTypeAnnotation")
  .bases("Type");

def("StringTypeAnnotation")
  .bases("Type");

def("StringLiteralTypeAnnotation")
  .bases("Type")
  .build("value", "raw")
  .field("value", isString)
  .field("raw", isString);

def("BooleanTypeAnnotation")
  .bases("Type");

def("TypeAnnotation")
  .bases("Node")
  .build("typeAnnotation")
  .field("typeAnnotation", def("Type"));

def("NullableTypeAnnotation")
  .bases("Type")
  .build("typeAnnotation")
  .field("typeAnnotation", def("Type"));

def("FunctionTypeAnnotation")
  .bases("Type")
  .build("params", "returnType", "rest", "typeParameters")
  .field("params", [def("FunctionTypeParam")])
  .field("returnType", def("Type"))
  .field("rest", or(def("FunctionTypeParam"), null))
  .field("typeParameters", or(def("TypeParameterDeclaration"), null));

def("FunctionTypeParam")
  .bases("Node")
  .build("name", "typeAnnotation", "optional")
  .field("name", def("Identifier"))
  .field("typeAnnotation", def("Type"))
  .field("optional", isBoolean);
  
def("ArrayTypeAnnotation")
  .bases("Type")
  .build("elementType")
  .field("elementType", def("Type"));

def("ObjectTypeAnnotation")
  .bases("Type")
  .build("properties")
  .field("properties", [def("ObjectTypeProperty")])
  .field("indexers", [def("ObjectTypeIndexer")], defaults.emptyArray)
  .field("callProperties", [def("ObjectTypeCallProperty")], defaults.emptyArray);

def("ObjectTypeProperty")
  .bases("Node")
  .build("key", "value", "optional")
  .field("key", or(def("Literal"), def("Identifier")))
  .field("value", def("Type"))
  .field("optional", isBoolean);

def("ObjectTypeIndexer")
  .bases("Node")
  .build("id", "key", "value")
  .field("id", def("Identifier"))
  .field("key", def("Type"))
  .field("value", def("Type"));

def("ObjectTypeCallProperty")
  .bases("Node")
  .build("value")
  .field("value", def("FunctionTypeAnnotation"))
  .field("static", isBoolean, false);

def("QualifiedTypeIdentifier")
  .bases("Node")
  .build("qualification", "id")
  .field("qualification", or(def("Identifier"), def("QualifiedTypeIdentifier")))
  .field("id", def("Identifier"));

def("GenericTypeAnnotation")
  .bases("Type")
  .build("id", "typeParameters")
  .field("id", or(def("Identifier"), def("QualifiedTypeIdentifier")))
  .field("typeParameters", or(def("TypeParameterInstantiation"), null));

def("MemberTypeAnnotation")
  .bases("Type")
  .build("object", "property")
  .field("object", def("Identifier"))
  .field("property", or(def("MemberTypeAnnotation"), def("GenericTypeAnnotation")));

def("UnionTypeAnnotation")
  .bases("Type")
  .build("types")
  .field("types", [def("Type")]);

def("IntersectionTypeAnnotation")
  .bases("Type")
  .build("types")
  .field("types", [def("Type")]);

def("TypeofTypeAnnotation")
  .bases("Type")
  .build("argument")
  .field("argument", def("Type"));

def("Identifier")
  .field("typeAnnotation", or(def("TypeAnnotation"), null), defaults["null"]);

def("TypeParameterDeclaration")
  .bases("Node")
  .build("params")
  .field("params", [def("Identifier")]);

def("TypeParameterInstantiation")
  .bases("Node")
  .build("params")
  .field("params", [def("Type")]);

def("Function")
  .field("returnType", or(def("TypeAnnotation"), null), defaults["null"])
  .field("typeParameters", or(def("TypeParameterDeclaration"), null), defaults["null"]);

def("ClassProperty")
  .build("key", "typeAnnotation")
  .field("typeAnnotation", def("TypeAnnotation"))
  .field("static", isBoolean, false);

def("ClassImplements")
  .field("typeParameters", or(def("TypeParameterInstantiation"), null), defaults["null"]);

def("InterfaceDeclaration")
  .bases("Statement")
  .build("id", "body", "extends")
  .field("id", def("Identifier"))
  .field("typeParameters", or(def("TypeParameterDeclaration"), null), defaults["null"])
  .field("body", def("ObjectTypeAnnotation"))
  .field("extends", [def("InterfaceExtends")]);

def("InterfaceExtends")
  .bases("Node")
  .build("id")
  .field("id", def("Identifier"))
  .field("typeParameters", or(def("TypeParameterInstantiation"), null));

def("TypeAlias")
  .bases("Statement")
  .build("id", "typeParameters", "right")
  .field("id", def("Identifier"))
  .field("typeParameters", or(def("TypeParameterDeclaration"), null))
  .field("right", def("Type"));
  
def("TypeCastExpression")
  .bases("Expression")
  .build("expression", "typeAnnotation")
  .field("expression", def("Expression"))
  .field("typeAnnotation", def("TypeAnnotation"));

def("TupleTypeAnnotation")
  .bases("Type")
  .build("types")
  .field("types", [def("Type")]);

def("DeclareVariable")
  .bases("Statement")
  .build("id")
  .field("id", def("Identifier"));

def("DeclareFunction")
  .bases("Statement")
  .build("id")
  .field("id", def("Identifier"));

def("DeclareClass")
  .bases("InterfaceDeclaration")
  .build("id");

def("DeclareModule")
  .bases("Statement")
  .build("id", "body")
  .field("id", or(def("Identifier"), def("Literal")))
  .field("body", def("BlockStatement"));
