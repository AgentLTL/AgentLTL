"""
agentltl/_parser.py – Parse LTL formula strings into AST nodes.

Supported grammar (case-insensitive keywords)::

    formula  ::= impl
    impl     ::= disj ('->' impl)?
    disj     ::= conj ('|' conj)*
    conj     ::= unary ('&' unary)*
    unary    ::= '!' unary
               | 'G' unary         -- Globally
               | 'F' unary         -- Eventually
               | 'X' unary         -- Next
               | binary
    binary   ::= atom ('U' unary)?  -- Until
               | atom ('W' unary)?  -- WeakUntil
               | atom ('R' unary)?  -- Release
    atom     ::= 'true' | 'false'
               | function_call
               | '(' formula ')'
    function_call
             ::= 'called'       '(' STRING ')'
               | 'before'       '(' STRING ',' STRING ')'
               | 'after'        '(' STRING ',' STRING ')'
               | 'eventually'   '(' STRING ')'
               | 'always'       '(' STRING ')'
               | 'all_before'   '(' '[' STRING (',' STRING)* ']' ',' STRING ')'
               | 'branch_called'  '(' STRING (',' STRING)? ')'

    STRING   ::= '"' [^"]* '"'

Example::

    >>> from agentltl import parse
    >>> f = parse('before("a", "b") & F(called("c"))')
"""

from __future__ import annotations

import re
from typing import List, Optional

from ._ast import (
    After,
    AllBefore,
    And,
    Before,
    BranchCalled,
    Called,
    Eventually,
    Formula,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Release,
    Until,
    WeakUntil,
)


# ─────────────────────────────────────────────────────────────────────────────
# Token types
# ─────────────────────────────────────────────────────────────────────────────

class _TokenType:
    STRING = "STRING"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACK = "LBRACK"
    RBRACK = "RBRACK"
    COMMA = "COMMA"
    BANG = "BANG"
    AMP = "AMP"
    PIPE = "PIPE"
    ARROW = "ARROW"
    IDENT = "IDENT"
    EOF = "EOF"


class _Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type_: str, value: str, pos: int):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, pos={self.pos})"


# ─────────────────────────────────────────────────────────────────────────────
# Lexer
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_SPEC = [
    ("STRING",  r'"[^"]*"'),
    ("ARROW",   r"->"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("LBRACK",  r"\["),
    ("RBRACK",  r"\]"),
    ("COMMA",   r","),
    ("BANG",    r"!"),
    ("AMP",     r"&"),
    ("PIPE",    r"\|"),
    ("IDENT",   r"[A-Za-z_][A-Za-z_0-9]*"),
    ("SKIP",    r"[ \t\n]+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))


def _tokenize(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP":
            continue
        if kind == "STRING":
            value = value[1:-1]  # strip quotes
        tokens.append(_Token(kind, value, m.start()))
    tokens.append(_Token(_TokenType.EOF, "", len(text)))
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class _Parser:
    """Recursive-descent parser for LTL formulas."""

    def __init__(self, tokens: List[_Token], source: str):
        self.tokens = tokens
        self.source = source
        self.pos = 0

    def _peek(self) -> _Token:
        return self.tokens[self.pos]

    def _advance(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, type_: str) -> _Token:
        tok = self._advance()
        if tok.type != type_:
            raise SyntaxError(
                f"Expected {type_} but got {tok.type} ({tok.value!r}) "
                f"at position {tok.pos} in: {self.source!r}"
            )
        return tok

    def _match(self, type_: str) -> Optional[_Token]:
        if self._peek().type == type_:
            return self._advance()
        return None

    def parse(self) -> Formula:
        f = self._parse_impl()
        if self._peek().type != _TokenType.EOF:
            tok = self._peek()
            raise SyntaxError(
                f"Unexpected token {tok.type} ({tok.value!r}) "
                f"at position {tok.pos} in: {self.source!r}"
            )
        return f

    def _parse_impl(self) -> Formula:
        left = self._parse_disj()
        if self._match(_TokenType.ARROW):
            right = self._parse_impl()
            return Implies(left, right)
        return left

    def _parse_disj(self) -> Formula:
        left = self._parse_conj()
        while self._match(_TokenType.PIPE):
            right = self._parse_conj()
            left = Or(left, right)
        return left

    def _parse_conj(self) -> Formula:
        left = self._parse_unary()
        while self._match(_TokenType.AMP):
            right = self._parse_unary()
            left = And(left, right)
        return left

    def _parse_unary(self) -> Formula:
        tok = self._peek()
        if tok.type == _TokenType.BANG:
            self._advance()
            return Not(self._parse_unary())

        if tok.type == _TokenType.IDENT:
            upper = tok.value.upper()
            if upper == "G":
                self._advance()
                return Globally(self._parse_unary())
            if upper == "F":
                self._advance()
                return Eventually(self._parse_unary())
            if upper == "X":
                self._advance()
                return Next(self._parse_unary())

        return self._parse_binary()

    def _parse_binary(self) -> Formula:
        left = self._parse_atom()
        tok = self._peek()
        if tok.type == _TokenType.IDENT:
            upper = tok.value.upper()
            if upper == "U":
                self._advance()
                right = self._parse_unary()
                return Until(left, right)
            if upper == "W":
                self._advance()
                right = self._parse_unary()
                return WeakUntil(left, right)
            if upper == "R":
                self._advance()
                right = self._parse_unary()
                return Release(left, right)
        return left

    def _parse_atom(self) -> Formula:
        tok = self._peek()

        if tok.type == _TokenType.LPAREN:
            self._advance()
            f = self._parse_impl()
            self._expect(_TokenType.RPAREN)
            return f

        if tok.type == _TokenType.IDENT and tok.value.lower() == "true":
            self._advance()
            from ._ast import Predicate
            return Predicate(fn=lambda t, p: True, description="true")

        if tok.type == _TokenType.IDENT and tok.value.lower() == "false":
            self._advance()
            from ._ast import Predicate
            return Predicate(fn=lambda t, p: False, description="false")

        if tok.type == _TokenType.IDENT:
            return self._parse_function_call()

        raise SyntaxError(
            f"Unexpected token {tok.type} ({tok.value!r}) "
            f"at position {tok.pos} in: {self.source!r}"
        )

    def _parse_function_call(self) -> Formula:
        name_tok = self._advance()
        name = name_tok.value.lower()

        self._expect(_TokenType.LPAREN)

        if name == "called":
            tool = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return Called(tool)

        if name == "before":
            a = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.COMMA)
            b = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return Before(a, b)

        if name == "after":
            a = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.COMMA)
            b = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return After(a, b)

        if name in ("eventually", "always"):
            arg = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            if name == "eventually":
                return Eventually(Called(arg))
            return Globally(Called(arg))

        if name == "all_before":
            self._expect(_TokenType.LBRACK)
            tools: List[str] = [self._expect(_TokenType.STRING).value]
            while self._match(_TokenType.COMMA):
                if self._peek().type == _TokenType.STRING:
                    tools.append(self._expect(_TokenType.STRING).value)
                else:
                    break
            if self._peek().type == _TokenType.RBRACK:
                self._advance()
                self._expect(_TokenType.COMMA)
            target = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return AllBefore(tools, target)

        if name == "branch_called":
            correct = self._expect(_TokenType.STRING).value
            wrong = None
            if self._match(_TokenType.COMMA):
                wrong = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return BranchCalled(correct, wrong)

        raise SyntaxError(
            f"Unknown function '{name_tok.value}' "
            f"at position {name_tok.pos} in: {self.source!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse(text: str) -> Formula:
    """Parse an LTL formula string into an AST :class:`Formula` node.

    Examples::

        >>> parse('before("a", "b")')
        Before(a='a', b='b')

        >>> parse('called("x") & F(called("y"))')
        And(left=Called(tool='x'), right=Eventually(operand=Called(tool='y')))
    """
    tokens = _tokenize(text)
    parser = _Parser(tokens, text)
    return parser.parse()
