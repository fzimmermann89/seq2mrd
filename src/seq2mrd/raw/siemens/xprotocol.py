"""Siemens XProtocol parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

Scalar: TypeAlias = str | int | float

MAP_NODE_TYPES = {'ParamMap', 'Pipe', 'PipeService', 'ParamFunctor'}
OPAQUE_BLOCK_TAG_NAMES = {'Dependency', 'ProtocolComposer'}


class XProtocolParseError(ValueError):
    """Raised when Siemens XProtocol text cannot be parsed."""


@dataclass(slots=True)
class XProtocolArrayValue:
    """One nested Siemens array payload."""

    values: list[Scalar] = field(default_factory=list)
    children: list[XProtocolArrayValue] = field(default_factory=list)


@dataclass(slots=True)
class XProtocolNode:
    """One Siemens XProtocol node."""

    name: str
    node_type: str
    values: list[Scalar] = field(default_factory=list)
    children: list[XProtocolNode] = field(default_factory=list)
    properties: dict[str, list[Scalar]] = field(default_factory=dict)
    default: XProtocolNode | None = None
    array_values: list[XProtocolArrayValue] = field(default_factory=list)

    def child(self, name: str) -> XProtocolNode | None:
        """Return the first child with the requested name.

        Parameters
        ----------
        name
            Child node name.

        Returns
        -------
            Matching child node, if available.
        """
        normalized_name = name.casefold()
        for child in self.expanded_children():
            if child.name.casefold() == normalized_name:
                return child
        for property_name, property_values in self.properties.items():
            if property_name.casefold() == normalized_name:
                return XProtocolNode(name=property_name, node_type='Property', values=list(property_values))
        return None

    def child_index(self, index: int) -> XProtocolNode | None:
        """Return a child by index.

        Parameters
        ----------
        index
            Child index.

        Returns
        -------
            Matching child node, if available.
        """
        expanded_children = self.expanded_children()
        if 0 <= index < len(expanded_children):
            return expanded_children[index]
        return None

    def expanded_children(self) -> list[XProtocolNode]:
        """Expand Siemens ParamArray items on first access.

        Returns
        -------
            Materialized child nodes.
        """
        if self.node_type != 'ParamArray':
            return self.children
        if self.children or self.default is None:
            return self.children
        self.children = [materialize_array_item(self.default, array_value) for array_value in self.array_values]
        return self.children


@dataclass(slots=True)
class SiemensHeader:
    """Queryable Siemens XProtocol header."""

    root: XProtocolNode

    def exists(self, path: str) -> bool:
        """Check whether a Siemens header path exists.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.

        Returns
        -------
            True when the path resolves to a node.
        """
        return self.resolve(path) is not None

    def get(self, path: str, default: Scalar | None = None) -> Scalar | None:
        """Return the first scalar value at a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.
        default
            Value returned when the path is missing.

        Returns
        -------
            First scalar value at the resolved path.
        """
        resolved_node = self.resolve(path)
        if resolved_node is None or not resolved_node.values:
            return default
        return resolved_node.values[0]

    def get_str(self, path: str, default: str | None = None) -> str | None:
        """Return a string value at a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.
        default
            Value returned when the path is missing.

        Returns
        -------
            String value for the resolved path.
        """
        value = self.get(path)
        if value is None:
            return default
        return str(value)

    def get_int(self, path: str, default: int | None = None) -> int | None:
        """Return an integer value at a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.
        default
            Value returned when the path is missing.

        Returns
        -------
            Integer value for the resolved path.
        """
        value = self.get(path)
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return int(str(value))

    def get_float(self, path: str, default: float | None = None) -> float | None:
        """Return a float value at a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.
        default
            Value returned when the path is missing.

        Returns
        -------
            Float value for the resolved path.
        """
        value = self.get(path)
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))

    def get_list(self, path: str, default: list[Scalar] | None = None) -> list[Scalar]:
        """Return all scalar values at a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.
        default
            Values returned when the path is missing.

        Returns
        -------
            All scalar values for the resolved path.
        """
        resolved_node = self.resolve(path)
        if resolved_node is None:
            return [] if default is None else list(default)
        if resolved_node.values:
            return list(resolved_node.values)
        if resolved_node.node_type == 'ParamArray':
            values: list[Scalar] = []
            for child in resolved_node.expanded_children():
                values.extend(child.values)
            return values
        return [] if default is None else list(default)

    def require(self, path: str) -> Scalar:
        """Return a required value or raise `KeyError`.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.

        Returns
        -------
            Required value for the resolved path.
        """
        value = self.get(path)
        if value is None:
            raise KeyError(path)
        return value

    def resolve(self, path: str) -> XProtocolNode | None:
        """Resolve a Siemens header path.

        Parameters
        ----------
        path
            Dot-separated Siemens header path.

        Returns
        -------
            Resolved node, if available.
        """
        current_node = self.root
        path_components = [component for component in normalize_path(path).split('.') if component]
        if path_components and path_components[0].casefold() == current_node.name.casefold():
            path_components = path_components[1:]
        for path_component in path_components:
            if not path_component:
                continue
            if path_component.isdigit():
                value_index = int(path_component)
                if current_node.values:
                    if not 0 <= value_index < len(current_node.values):
                        return None
                    current_node = XProtocolNode(
                        name=current_node.name,
                        node_type='ScalarValue',
                        values=[current_node.values[value_index]],
                    )
                    continue
                current_node = current_node.child_index(value_index)  # type: ignore[assignment]
            else:
                current_node = current_node.child(path_component)
            if current_node is None:
                return None
        return current_node


@dataclass(slots=True)
class ParsedEntry:
    """Intermediate parsed XProtocol entry."""

    node: XProtocolNode | None = None
    properties: dict[str, list[Scalar]] = field(default_factory=dict)


def parse_siemens_header(text: str) -> SiemensHeader:
    """Parse Siemens XProtocol text into a queryable header tree.

    Parameters
    ----------
    text
        Siemens XProtocol text.

    Returns
    -------
        Queryable Siemens header.
    """
    root_node = XProtocolParser(text).parse()
    return SiemensHeader(root_node)


class XProtocolParser:
    """Recursive-descent parser for Siemens XProtocol text."""

    def __init__(self, text: str) -> None:
        """Initialize the parser.

        Parameters
        ----------
        text
            Siemens XProtocol text.
        """
        self.text = text
        self.length = len(text)
        self.position = 0

    def parse(self) -> XProtocolNode:
        """Parse the full XProtocol document.

        Returns
        -------
            Parsed root node.
        """
        self.skip_whitespace()
        self.expect_literal('<XProtocol>')
        self.skip_whitespace()
        self.expect('{')
        children, properties = self.parse_entries(until='}')
        self.expect('}')
        if not children:
            raise XProtocolParseError('XProtocol contains no root child.')
        root_node = children[0]
        for property_name, property_values in properties.items():
            root_node.properties.setdefault(property_name, []).extend(property_values)
        return root_node

    def parse_entries(self, until: str) -> tuple[list[XProtocolNode], dict[str, list[Scalar]]]:
        """Parse entries until a terminator is reached.

        Parameters
        ----------
        until
            Terminator character.

        Returns
        -------
            Parsed child nodes and metadata properties.
        """
        children: list[XProtocolNode] = []
        properties: dict[str, list[Scalar]] = {}
        while True:
            self.skip_whitespace()
            if self.peek() == until:
                return children, properties
            if self.peek() != '<':
                raise XProtocolParseError(f'Expected tag at position {self.position}.')
            entry = self.parse_entry()
            if entry.node is not None:
                children.append(entry.node)
            for property_name, property_values in entry.properties.items():
                properties.setdefault(property_name, []).extend(property_values)

    def parse_entry(self) -> ParsedEntry:
        """Parse one tagged entry.

        Returns
        -------
            Parsed entry.
        """
        tag_name, tag_argument = self.parse_tag_header()
        self.skip_whitespace()
        if self.peek() != '{':
            values = self.parse_inline_scalar_values()
            return self.make_entry(tag_name, tag_argument, values, {})

        self.expect('{')
        if tag_name in MAP_NODE_TYPES:
            children, properties = self.parse_entries(until='}')
            self.expect('}')
            return ParsedEntry(
                node=XProtocolNode(
                    name=tag_argument,
                    node_type=tag_name,
                    children=children,
                    properties=properties,
                )
            )

        if tag_name == 'ParamArray':
            return ParsedEntry(node=self.parse_param_array(tag_argument))

        if tag_name in OPAQUE_BLOCK_TAG_NAMES:
            raw_text = self.parse_raw_block()
            values: list[Scalar] = [raw_text] if raw_text else []
            return self.make_entry(tag_name, tag_argument, values, {})

        values, properties = self.parse_value_block_contents()
        self.expect('}')
        return self.make_entry(tag_name, tag_argument, values, properties)

    def parse_param_array(self, name: str) -> XProtocolNode:
        """Parse a Siemens `ParamArray`.

        Parameters
        ----------
        name
            Node name.

        Returns
        -------
            Parsed ParamArray node.
        """
        default_node: XProtocolNode | None = None
        array_values: list[XProtocolArrayValue] = []
        properties: dict[str, list[Scalar]] = {}
        while True:
            self.skip_whitespace()
            next_character = self.peek()
            if next_character == '}':
                self.expect('}')
                return XProtocolNode(
                    name=name,
                    node_type='ParamArray',
                    default=default_node,
                    array_values=array_values,
                    properties=properties,
                )
            if next_character == '<':
                tag_name, tag_argument = self.parse_tag_header()
                self.skip_whitespace()
                if tag_name == 'Default':
                    if self.peek() == '{':
                        default_node = self.parse_default_node()
                    elif self.peek() == '<':
                        entry = self.parse_entry()
                        if entry.node is None:
                            raise XProtocolParseError('Default node unexpectedly parsed as metadata.')
                        default_node = entry.node
                    else:
                        raise XProtocolParseError('Expected default child node.')
                    continue
                if self.peek() == '{':
                    self.expect('{')
                    values, nested_properties = self.parse_value_block_contents()
                    self.expect('}')
                else:
                    values = self.parse_inline_scalar_values()
                    nested_properties = {}
                property_name = tag_argument if tag_argument != tag_name else tag_name
                if values:
                    properties.setdefault(property_name, []).extend(values)
                for nested_property_name, nested_property_values in nested_properties.items():
                    properties.setdefault(nested_property_name, []).extend(nested_property_values)
                continue
            if next_character == '{':
                array_values.append(self.parse_array_value())
                continue
            raise XProtocolParseError(f'Unexpected token in ParamArray at position {self.position}.')

    def parse_default_node(self) -> XProtocolNode:
        """Parse the default node stored inside a `ParamArray`.

        Returns
        -------
            Parsed default node.
        """
        self.skip_whitespace()
        self.expect('{')
        self.skip_whitespace()
        if self.peek() != '<':
            raise XProtocolParseError('Expected default child node.')
        entry = self.parse_entry()
        self.skip_whitespace()
        self.expect('}')
        if entry.node is None:
            raise XProtocolParseError('Default node unexpectedly parsed as metadata.')
        return entry.node

    def parse_value_block_contents(self) -> tuple[list[Scalar], dict[str, list[Scalar]]]:
        """Parse values and nested properties inside a value block.

        Returns
        -------
            Parsed scalar values and metadata properties.
        """
        values: list[Scalar] = []
        properties: dict[str, list[Scalar]] = {}
        while True:
            self.skip_whitespace()
            next_character = self.peek()
            if next_character == '}':
                return values, properties
            if next_character == '<':
                entry = self.parse_entry()
                if entry.node is not None:
                    properties.setdefault(entry.node.name, []).extend(entry.node.values)
                    for property_name, property_values in entry.node.properties.items():
                        properties.setdefault(property_name, []).extend(property_values)
                for property_name, property_values in entry.properties.items():
                    properties.setdefault(property_name, []).extend(property_values)
                continue
            if next_character == '{':
                values.extend(self.parse_anonymous_block_values())
                continue
            values.append(self.parse_scalar())

    def parse_array_value(self) -> XProtocolArrayValue:
        """Parse one Siemens array payload block.

        Returns
        -------
            Parsed array payload.
        """
        self.skip_whitespace()
        self.expect('{')
        values: list[Scalar] = []
        children: list[XProtocolArrayValue] = []
        while True:
            self.skip_whitespace()
            next_character = self.peek()
            if next_character == '}':
                self.expect('}')
                return XProtocolArrayValue(values=values, children=children)
            if next_character == '{':
                children.append(self.parse_array_value())
                continue
            if next_character == '<':
                entry = self.parse_entry()
                if entry.node is not None and entry.node.values:
                    children.append(XProtocolArrayValue(values=list(entry.node.values)))
                for property_values in entry.properties.values():
                    if property_values:
                        children.append(XProtocolArrayValue(values=list(property_values)))
                continue
            values.append(self.parse_scalar())

    def parse_anonymous_block_values(self) -> list[Scalar]:
        """Parse scalar values from an anonymous block.

        Returns
        -------
            Parsed scalar values.
        """
        self.skip_whitespace()
        self.expect('{')
        values, _properties = self.parse_value_block_contents()
        self.expect('}')
        return values

    def parse_raw_block(self) -> str:
        """Consume an opaque block and return its raw text.

        Returns
        -------
            Raw block text without outer braces.
        """
        start_position = self.position
        depth = 1
        while depth:
            next_character = self.next_character()
            if next_character == '{':
                depth += 1
            elif next_character == '}':
                depth -= 1
        return self.text[start_position : self.position - 1].strip()

    def make_entry(
        self,
        tag_name: str,
        tag_argument: str,
        values: list[Scalar],
        properties: dict[str, list[Scalar]],
    ) -> ParsedEntry:
        """Build a parsed entry from a tag payload.

        Parameters
        ----------
        tag_name
            Siemens tag name.
        tag_argument
            Siemens tag argument.
        values
            Parsed scalar values.
        properties
            Parsed metadata properties.

        Returns
        -------
            Parsed entry.
        """
        if tag_argument == tag_name:
            merged_properties = {tag_name: list(values)}
            for property_name, property_values in properties.items():
                merged_properties.setdefault(property_name, []).extend(property_values)
            return ParsedEntry(properties=merged_properties)
        return ParsedEntry(
            node=XProtocolNode(
                name=tag_argument,
                node_type=tag_name,
                values=list(values),
                properties={name: list(property_values) for name, property_values in properties.items()},
            )
        )

    def parse_inline_scalar_values(self) -> list[Scalar]:
        """Parse scalar values that follow a tag without a block.

        Returns
        -------
            Parsed scalar values.
        """
        values: list[Scalar] = []
        while True:
            self.skip_inline_whitespace()
            next_character = self.peek()
            if next_character in {'', '\r', '\n', '<', '}'}:
                return values
            values.append(self.parse_scalar())

    def parse_tag_header(self) -> tuple[str, str]:
        """Parse one Siemens tag header.

        Returns
        -------
            Tag name and tag argument.
        """
        self.expect('<')
        tag_name = self.read_until(stop_characters={'.', '>'})
        if self.peek() == '>':
            self.expect('>')
            return tag_name, tag_name
        self.expect('.')
        tag_argument = self.parse_quoted_string()
        self.expect('>')
        return tag_name, tag_argument

    def parse_scalar(self) -> Scalar:
        """Parse one scalar token.

        Returns
        -------
            Parsed scalar value.
        """
        self.skip_whitespace()
        if self.peek() == '"':
            return self.parse_quoted_string()
        start_position = self.position
        while self.position < self.length and self.text[self.position] not in ' \t\r\n{}<>':
            self.position += 1
        token = self.text[start_position : self.position]
        if not token:
            raise XProtocolParseError(f'Expected scalar at position {start_position}.')
        for parser in (int, float):
            try:
                return parser(token)
            except ValueError:
                continue
        return token

    def parse_quoted_string(self) -> str:
        """Parse one quoted string token.

        Returns
        -------
            Parsed string.
        """
        self.expect('"')
        start_position = self.position
        while self.position < self.length and self.text[self.position] != '"':
            self.position += 1
        if self.position >= self.length:
            raise XProtocolParseError('Unterminated quoted string.')
        value = self.text[start_position : self.position]
        self.expect('"')
        return value

    def read_until(self, stop_characters: set[str]) -> str:
        """Read characters until a stop character is encountered.

        Parameters
        ----------
        stop_characters
            Characters that terminate the read.

        Returns
        -------
            Read text.
        """
        start_position = self.position
        while self.position < self.length and self.text[self.position] not in stop_characters:
            self.position += 1
        if self.position == start_position:
            raise XProtocolParseError(f'Expected text at position {start_position}.')
        return self.text[start_position : self.position]

    def expect_literal(self, literal: str) -> None:
        """Consume an expected literal token.

        Parameters
        ----------
        literal
            Literal token.
        """
        if not self.text.startswith(literal, self.position):
            raise XProtocolParseError(f'Expected {literal!r} at position {self.position}.')
        self.position += len(literal)

    def expect(self, character: str) -> None:
        """Consume an expected single character.

        Parameters
        ----------
        character
            Character to consume.
        """
        self.skip_whitespace()
        if self.peek() != character:
            raise XProtocolParseError(f'Expected {character!r} at position {self.position}.')
        self.position += 1

    def peek(self) -> str:
        """Return the next character without consuming it.

        Returns
        -------
            Next character or an empty string at end of input.
        """
        if self.position >= self.length:
            return ''
        return self.text[self.position]

    def next_character(self) -> str:
        """Consume and return the next character.

        Returns
        -------
            Consumed character.
        """
        if self.position >= self.length:
            raise XProtocolParseError('Unexpected end of input.')
        next_character = self.text[self.position]
        self.position += 1
        return next_character

    def skip_whitespace(self) -> None:
        """Skip ASCII whitespace."""
        while self.position < self.length and self.text[self.position] in ' \t\r\n':
            self.position += 1

    def skip_inline_whitespace(self) -> None:
        """Skip horizontal whitespace without crossing line boundaries."""
        while self.position < self.length and self.text[self.position] in ' \t':
            self.position += 1


def normalize_path(path: str) -> str:
    """Normalize Siemens path aliases across software lines.

    Parameters
    ----------
    path
        Dot-separated Siemens header path.

    Returns
    -------
        Normalized path.
    """
    return path.replace('.sWiPMemBlock.', '.sWipMemBlock.')


def materialize_array_item(template: XProtocolNode, item: XProtocolArrayValue) -> XProtocolNode:
    """Instantiate one concrete array element from the default template.

    Parameters
    ----------
    template
        Default array template node.
    item
        Array payload values.

    Returns
    -------
        Materialized array element.
    """
    node = clone_node(template)
    apply_array_item(node, item)
    return node


def clone_node(node: XProtocolNode) -> XProtocolNode:
    """Deep-copy an XProtocol node.

    Parameters
    ----------
    node
        Node to clone.

    Returns
    -------
        Cloned node.
    """
    return XProtocolNode(
        name=node.name,
        node_type=node.node_type,
        values=list(node.values),
        children=[clone_node(child) for child in node.children],
        properties={key: list(value) for key, value in node.properties.items()},
        default=clone_node(node.default) if node.default is not None else None,
        array_values=[clone_array_value(value) for value in node.array_values],
    )


def clone_array_value(node: XProtocolArrayValue) -> XProtocolArrayValue:
    """Deep-copy an intermediate array payload.

    Parameters
    ----------
    node
        Array payload to clone.

    Returns
    -------
        Cloned array payload.
    """
    return XProtocolArrayValue(
        values=list(node.values),
        children=[clone_array_value(child) for child in node.children],
    )


def apply_array_item(node: XProtocolNode, item: XProtocolArrayValue) -> None:
    """Apply array payload values to a cloned default node.

    Parameters
    ----------
    node
        Cloned default node.
    item
        Array payload values.
    """
    if node.node_type in MAP_NODE_TYPES:
        if len(item.children) == 1 and not item.values and item.children[0].children and len(node.children) != len(item.children):
            item = item.children[0]
        for child, child_item in zip(node.children, item.children, strict=False):
            apply_array_item(child, child_item)
        return
    if node.node_type == 'ParamArray':
        node.values.clear()
        node.children.clear()
        node.array_values = [clone_array_value(value) for value in item.children]
        return
    node.values = list(item.values)
