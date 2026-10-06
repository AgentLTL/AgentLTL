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
               | 'Y' unary         -- Previous (past)
               | 'O' unary         -- Once (past)
               | 'H' unary         -- Historically (past)
               | binary
    binary   ::= atom ('U' unary)?  -- Until
               | atom ('W' unary)?  -- WeakUntil
               | atom ('R' unary)?  -- Release
               | atom ('S' unary)?  -- Since (past)
    atom     ::= 'true' | 'false'
               | function_call
               | '(' formula ')'
    function_call
             ::= 'called'       '(' STRING ')'
               | 'now'          '(' STRING ')'   -- the call at this step
               | 'before'       '(' STRING ',' STRING ')'
               | 'after'        '(' STRING ',' STRING ')'
               | 'eventually'   '(' STRING ')'
               | 'always'       '(' STRING ')'
               | 'all_before'   '(' '[' STRING (',' STRING)* ']' ',' STRING ')'
               | 'branch_called'  '(' STRING (',' STRING)? ')'
               | 'called_with'  '(' STRING (',' IDENT '=' value)* ')'
               | 'called_n'     '(' STRING ',' CMP NUMBER ')'      -- called_n("x", <= 2)
               | 'in_order'     '(' '[' STRING (',' STRING)* ']' ')'
               | 'within_steps' '(' STRING ',' STRING ',' NUMBER ')'
               | 'instance_before' '(' STRING '[' NUMBER ']' ',' STRING '[' NUMBER ']' ')'
               | 'count_before' '(' formula ',' CMP NUMBER ')'     -- past positions where it held

    value    ::= STRING | NUMBER | 'true' | 'false' | 'null'
    CMP      ::= '==' | '>=' | '<=' | '>' | '<'
    STRING   ::= '"' [^"]* '"'

``str(formula)`` prints this syntax, so ``parse(str(f)) == f`` for every formula built
from these nodes (predicates, call patterns and quantifiers hold Python callables and
cannot be written as text).

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
    CalledInOrder,
    CalledNTimes,
    CalledWith,
    CountBefore,
    InstanceBefore,
    WithinSteps,
    Eventually,
    Formula,
    Globally,
    Historically,
    Implies,
    Next,
    Not,
    Now,
    Once,
    Or,
    Previous,
    Release,
    Since,
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
    CMP = "CMP"
    EQ = "EQ"
    NUMBER = "NUMBER"
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
    ("CMP",     r"<=|>=|==|<|>"),
    ("EQ",      r"="),
    ("NUMBER",  r"-?\d+(?:\.\d+)?"),
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
    ("ERROR",   r"."),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))


def _tokenize(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP":
            continue
        if kind == "ERROR":
            raise SyntaxError(f"Unexpected character {value!r} at position {m.start()} in: {text!r}")
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
            if upper in ("Y", "O", "H") and self._peek_next_is_operand():
                self._advance()
                cls = {"Y": Previous, "O": Once, "H": Historically}[upper]
                return cls(self._parse_unary())

        return self._parse_binary()

    def _value(self):
        tok = self._advance()
        if tok.type == _TokenType.STRING:
            return tok.value
        if tok.type == _TokenType.NUMBER:
            return float(tok.value) if "." in tok.value else int(tok.value)
        if tok.type == _TokenType.IDENT and tok.value in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[tok.value]
        raise SyntaxError(f"Expected a value at position {tok.pos} in: {self.source!r}")

    def _string_list(self) -> List[str]:
        self._expect(_TokenType.LBRACK)
        items = [self._expect(_TokenType.STRING).value]
        while self._match(_TokenType.COMMA):
            items.append(self._expect(_TokenType.STRING).value)
        self._expect(_TokenType.RBRACK)
        return items

    def _indexed(self):
        tool = self._expect(_TokenType.STRING).value
        self._expect(_TokenType.LBRACK)
        n = int(self._expect(_TokenType.NUMBER).value)
        self._expect(_TokenType.RBRACK)
        return tool, n

    def _peek_next_is_operand(self) -> bool:
        """``O(...)``, ``H now(...)``: the letter is an operator, not a function name."""
        nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
        return nxt is not None and nxt.type in (_TokenType.LPAREN, _TokenType.IDENT,
                                                _TokenType.BANG)

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
            if upper == "S":
                self._advance()
                right = self._parse_unary()
                return Since(left, right)
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

        if name == "now":
            tool = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.RPAREN)
            return Now(tool)

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

        if name == "called_with":
            tool = self._expect(_TokenType.STRING).value
            args = {}
            while self._match(_TokenType.COMMA):
                key = self._expect(_TokenType.IDENT).value
                self._expect(_TokenType.EQ)
                args[key] = self._value()
            self._expect(_TokenType.RPAREN)
            return CalledWith(tool, args)

        if name == "called_n":
            tool = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.COMMA)
            op = self._expect(_TokenType.CMP).value
            n = int(self._expect(_TokenType.NUMBER).value)
            self._expect(_TokenType.RPAREN)
            return CalledNTimes(tool, n, op)

        if name == "in_order":
            tools = self._string_list()
            self._expect(_TokenType.RPAREN)
            return CalledInOrder(tools)

        if name == "within_steps":
            a = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.COMMA)
            b = self._expect(_TokenType.STRING).value
            self._expect(_TokenType.COMMA)
            n = int(self._expect(_TokenType.NUMBER).value)
            self._expect(_TokenType.RPAREN)
            return WithinSteps(a, b, n)

        if name == "instance_before":
            a, n = self._indexed()
            self._expect(_TokenType.COMMA)
            b, m = self._indexed()
            self._expect(_TokenType.RPAREN)
            return InstanceBefore(a, n, b, m)

        if name == "count_before":
            operand = self._parse_impl()
            self._expect(_TokenType.COMMA)
            op = self._expect(_TokenType.CMP).value
            n = int(self._expect(_TokenType.NUMBER).value)
            self._expect(_TokenType.RPAREN)
            return CountBefore(operand, n, op)

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
