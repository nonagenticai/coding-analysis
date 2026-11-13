"""Tests for processing/symbols.py and POST /api/analysis/symbols.

Every language sample asserts present names with kind AND parent, and a ghost
name that must be absent: a walker that reported every identifier would pass
a presence-only test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from hunt_agent import hunt
from processing.symbols import extract_symbols

GHOST = "ghostSymbolThatIsNotDefined"


def _has(result: dict, name: str, kind: str, parent: str | None = None) -> bool:
    return any(
        s["name"] == name and s["kind"] == kind and s["parent"] == parent for s in result["symbols"]
    )


def _names(result: dict) -> set[str]:
    return {s["name"] for s in result["symbols"]}


def _import(result: dict, module: str) -> dict:
    matches = [i for i in result["imports"] if i["module"] == module]
    assert matches, f"no import of {module}: {result['imports']}"
    return matches[0]


def _parse(code: str, **kwargs: str) -> dict:
    result = extract_symbols(code.encode(), **kwargs)
    assert result["parsed"] is True
    return result


PYTHON = """import os
import numpy as np
from collections import OrderedDict, defaultdict as dd
from . import sibling


def top(a):
    def inner():
        return 1
    return inner


class Box:
    def __init__(self):
        self.x = 1

    @staticmethod
    def make():
        return Box()

    class Inner:
        def deep(self):
            pass
"""

TYPESCRIPT = """import React, { useState as useS, useEffect } from "react";
import * as path from "path";

export const arrow = (y: string) => y;
const fnExpr = function (z: number) { return z; };

export function identity<T>(value: T): T {
  return value;
}

export interface Shape {
  area(): number;
}

export type Id = string | number;

export enum Color { Red, Green }

export abstract class Base {
  abstract describe(): string;
}

export class Circle extends Base implements Shape {
  handler = () => 1;
  constructor(private r: number) { super(); }
  area(): number { return Math.PI * this.r ** 2; }
  describe(): string { return "circle"; }
}

export default function () { return 0; }
"""

TSX_SAMPLE = """import React from "react";

export const Button = ({ label }: { label: string }) => <button>{label}</button>;

export function App() {
  return <div><Button label="x" /></div>;
}
"""

JAVASCRIPT = """const fs = require("fs");
const { join, resolve: res } = require("path");
import def, { named } from "./mod.js";

function* gen() { yield 1; }
async function load() { return fs.readFileSync("x"); }
const arrow = async (a) => a;
let expr = function named2() {};

class Widget {
  #secret() {}
  render() { return 1; }
  static create() { return new Widget(); }
}

module.exports = { Widget };
"""

GO = """package shapes

import (
\t"fmt"
\tm "math"
)

type Shape interface {
\tArea() float64
}

type Rect struct {
\tW, H float64
}

func (r *Rect) Area() float64 { return r.W * r.H }

func (r Rect) String() string { return fmt.Sprintf("%v", m.Pi) }

func New(w, h float64) *Rect { return &Rect{W: w, H: h} }
"""

JAVA = """package com.example;

import java.util.List;
import java.util.*;
import static java.lang.Math.max;

public class Service {
    private final int x;
    public Service(int x) { this.x = x; }
    public int run(List<String> items) { return max(x, items.size()); }
    static class Helper { void assist() {} }
    interface Callback { void done(); }
    enum Mode { ON, OFF }
    record Pair(int a, int b) {}
}
"""

RUST = """use std::collections::HashMap;
use std::io::{self, Read};
use crate::util::*;

pub mod inner {
    pub fn helper() -> u32 { 1 }
}

pub struct Counter { count: u32 }

pub enum State { Idle, Busy }

pub trait Tick {
    fn tick(&mut self);
}

impl Counter {
    pub fn new() -> Self { Counter { count: 0 } }
}

impl Tick for Counter {
    fn tick(&mut self) { self.count += 1; }
}

fn main() {}
"""

CPP = """#include <vector>
#include "widget.h"

namespace app {

class Widget {
public:
    int size() const;
    void inline_method() {}
};

struct Point { int x; int y; };

enum Color { Red, Green };

}  // namespace app

int app::Widget::size() const { return 0; }

static int helper(int a) { return a * 2; }

int main() { return helper(1); }
"""

CSHARP = """using System;
using System.Collections.Generic;
using Json = System.Text.Json;

namespace Demo.App
{
    public interface IRepo { void Save(); }
    public struct Point { public int X; }
    public enum Kind { A, B }
    public record Person(string Name);
    public class Repo : IRepo
    {
        public Repo() {}
        public void Save() {}
        private int Count() { int Local() => 1; return Local(); }
    }
}
"""


class TestPython:
    def test_definitions(self):
        r = _parse(PYTHON, file_path="mod.py")
        assert r["language"] == "python"
        assert r["parse_error_line"] is None
        assert _has(r, "top", "function")
        assert _has(r, "inner", "function")
        assert _has(r, "Box", "class")
        assert _has(r, "__init__", "method", "Box")
        assert _has(r, "make", "method", "Box")
        assert _has(r, "Inner", "class", "Box")
        assert _has(r, "deep", "method", "Inner")
        assert GHOST not in _names(r)
        assert not _has(r, "inner", "method", "Box")

    def test_imports(self):
        r = _parse(PYTHON, language="python")
        assert _import(r, "os")["names"] == []
        assert _import(r, "numpy")["names"] == ["np"]
        assert _import(r, "collections")["names"] == ["OrderedDict", "defaultdict"]
        assert _import(r, ".")["names"] == ["sibling"]


class TestTypeScript:
    def test_definitions(self):
        r = _parse(TYPESCRIPT, file_path="src/shapes.ts")
        assert r["language"] == "typescript"
        assert r["parse_error_line"] is None
        assert _has(r, "arrow", "const_function")
        assert _has(r, "fnExpr", "const_function")
        assert _has(r, "identity", "function")
        assert _has(r, "Shape", "interface")
        assert _has(r, "area", "method", "Shape")
        assert _has(r, "Id", "type")
        assert _has(r, "Color", "enum")
        assert _has(r, "Base", "class")
        assert _has(r, "describe", "method", "Base")
        assert _has(r, "Circle", "class")
        assert _has(r, "handler", "method", "Circle")
        assert _has(r, "constructor", "method", "Circle")
        assert _has(r, "area", "method", "Circle")
        assert _has(r, "describe", "method", "Circle")
        assert _has(r, "default", "function")
        assert GHOST not in _names(r)
        assert "y" not in _names(r)

    def test_imports(self):
        r = _parse(TYPESCRIPT, language="ts")
        assert _import(r, "react")["names"] == ["React", "useState", "useEffect"]
        assert _import(r, "path")["names"] == ["*"]

    def test_tsx_uses_tsx_grammar(self):
        r = _parse(TSX_SAMPLE, file_path="ui/App.tsx")
        assert r["language"] == "tsx"
        assert r["parse_error_line"] is None
        assert _has(r, "Button", "const_function")
        assert _has(r, "App", "function")
        assert GHOST not in _names(r)


class TestJavaScript:
    def test_definitions(self):
        r = _parse(JAVASCRIPT, file_path="lib/widget.js")
        assert r["language"] == "javascript"
        assert _has(r, "gen", "function")
        assert _has(r, "load", "function")
        assert _has(r, "arrow", "const_function")
        assert _has(r, "expr", "const_function")
        assert _has(r, "Widget", "class")
        assert _has(r, "#secret", "method", "Widget")
        assert _has(r, "render", "method", "Widget")
        assert _has(r, "create", "method", "Widget")
        assert GHOST not in _names(r)

    def test_require_and_import(self):
        r = _parse(JAVASCRIPT, language="javascript")
        assert _import(r, "fs")["names"] == ["fs"]
        assert "join" in _import(r, "path")["names"]
        assert _import(r, "./mod.js")["names"] == ["def", "named"]


class TestGo:
    def test_definitions(self):
        r = _parse(GO, file_path="shapes.go")
        assert _has(r, "Shape", "interface")
        assert _has(r, "Area", "method", "Shape")
        assert _has(r, "Rect", "struct")
        assert _has(r, "Area", "method", "Rect")
        assert _has(r, "String", "method", "Rect")
        assert _has(r, "New", "function")
        assert _has(r, "W", "field", "Rect")
        assert _has(r, "H", "field", "Rect")
        assert GHOST not in _names(r)
        assert not _has(r, "W", "variable", None)

    def test_imports(self):
        r = _parse(GO, language="golang")
        assert _import(r, "fmt")["names"] == []
        assert _import(r, "math")["names"] == ["m"]


class TestJava:
    def test_definitions(self):
        r = _parse(JAVA, file_path="Service.java")
        assert _has(r, "Service", "class")
        assert _has(r, "Service", "method", "Service")
        assert _has(r, "run", "method", "Service")
        assert _has(r, "Helper", "class", "Service")
        assert _has(r, "assist", "method", "Helper")
        assert _has(r, "Callback", "interface", "Service")
        assert _has(r, "done", "method", "Callback")
        assert _has(r, "Mode", "enum", "Service")
        assert _has(r, "Pair", "class", "Service")
        assert GHOST not in _names(r)

    def test_imports(self):
        r = _parse(JAVA, language="java")
        assert _import(r, "java.util.List")["names"] == ["List"]
        assert _import(r, "java.util")["names"] == ["*"]
        assert _import(r, "java.lang.Math.max")["names"] == ["max"]


class TestRust:
    def test_definitions(self):
        r = _parse(RUST, file_path="src/lib.rs")
        assert _has(r, "inner", "module")
        assert _has(r, "helper", "function")
        assert _has(r, "Counter", "struct")
        assert _has(r, "State", "enum")
        assert _has(r, "Tick", "trait")
        assert _has(r, "tick", "method", "Tick")
        assert _has(r, "new", "method", "Counter")
        assert _has(r, "tick", "method", "Counter")
        assert _has(r, "main", "function")
        assert GHOST not in _names(r)

    def test_imports(self):
        r = _parse(RUST, language="rust")
        assert _import(r, "std::collections")["names"] == ["HashMap"]
        assert _import(r, "std::io")["names"] == ["self", "Read"]
        assert _import(r, "crate::util")["names"] == ["*"]


class TestCpp:
    def test_definitions(self):
        r = _parse(CPP, file_path="widget.cpp")
        assert _has(r, "app", "namespace")
        assert _has(r, "Widget", "class")
        assert _has(r, "inline_method", "method", "Widget")
        assert _has(r, "Point", "struct")
        assert _has(r, "Color", "enum")
        assert _has(r, "helper", "function")
        assert _has(r, "main", "function")
        qualified = [s for s in r["symbols"] if s["name"] == "size" and s["line"] == 18]
        assert qualified, r["symbols"]
        assert qualified[0]["kind"] == "method"
        assert qualified[0]["parent"] == "Widget"
        assert GHOST not in _names(r)

    def test_includes(self):
        r = _parse(CPP, language="c++")
        assert {i["module"] for i in r["imports"]} == {"vector", "widget.h"}


class TestCSharp:
    def test_definitions(self):
        r = _parse(CSHARP, file_path="Repo.cs")
        assert r["language"] == "c_sharp"
        assert _has(r, "Demo.App", "namespace")
        assert _has(r, "IRepo", "interface")
        assert _has(r, "Save", "method", "IRepo")
        assert _has(r, "Point", "struct")
        assert _has(r, "Kind", "enum")
        assert _has(r, "Person", "class")
        assert _has(r, "Repo", "class")
        assert _has(r, "Repo", "method", "Repo")
        assert _has(r, "Save", "method", "Repo")
        assert _has(r, "Count", "method", "Repo")
        assert _has(r, "Local", "function")
        assert GHOST not in _names(r)

    def test_usings(self):
        r = _parse(CSHARP, language="csharp")
        assert _import(r, "System")["names"] == []
        assert _import(r, "System.Collections.Generic")["names"] == []
        assert _import(r, "System.Text.Json")["names"] == ["Json"]


CSHARP_MEMBERS = """class K {
    int Prop { get; set; }
    event EventHandler Ev, Ev2;
    event EventHandler Custom { add {} remove {} }
    public int this[int i] => i;
    public static K operator +(K a, K b) => a;
    public static implicit operator int(K k) => 0;
}
"""

TS_OBJECTS = """const obj = {
  op: () => 1,
  of: function () {},
  'quotedKey': () => 2,
  ['computedKey']: () => 3,
  [dynamicKey]: () => 4,
  notAFunction: 5,
};
register({ callback: () => 1 });
const C2 = class { cm() {} };
class Q { 'quoted'() {} ['comp']() {} [sym]() {} }
"""

JS_EXPORTS = """module.exports.mf = function () {};
exports.ef = () => 1;
module.exports = function named() {};
"""

CPP_EXTRA = """using Alias = int;
typedef int TD;
typedef int *PTD;
typedef void (*Callback)(int);
#define MAC(x) x
#define OBJ 1
void (*getfp())(int) { return 0; }
struct S { void (*cb)(int); int m(int); };
int f(int a);
void p(Foo);
void q(Foo *);
Widget w(config);
Widget w2(a, b);
Widget w3(42);
"""


class TestMissedDefinitions:
    """Definitions a validator found reported absent; each is a false absence."""

    def test_csharp_members(self):
        r = _parse(CSHARP_MEMBERS, language="c#")
        assert r["parse_error_line"] is None
        assert _has(r, "Prop", "property", "K")
        assert _has(r, "Ev", "event", "K")
        assert _has(r, "Ev2", "event", "K")
        assert _has(r, "Custom", "event", "K")
        assert _has(r, "this", "indexer", "K")
        assert _has(r, "operator +", "operator", "K")
        assert _has(r, "operator int", "operator", "K")
        assert GHOST not in _names(r)
        assert "i" not in _names(r)

    def test_ts_object_literal_methods_and_class_expressions(self):
        r = _parse(TS_OBJECTS, language="typescript")
        assert r["parse_error_line"] is None
        assert _has(r, "op", "method", "obj")
        assert _has(r, "of", "method", "obj")
        assert _has(r, "quotedKey", "method", "obj")
        assert _has(r, "computedKey", "method", "obj")
        assert _has(r, "callback", "method", None)
        assert _has(r, "C2", "class", None)
        assert _has(r, "cm", "method", "C2")
        assert _has(r, "quoted", "method", "Q")
        assert _has(r, "comp", "method", "Q")
        names = _names(r)
        assert GHOST not in names
        assert "notAFunction" not in names
        assert not any("dynamicKey" in n or "sym" in n for n in names)
        assert not any("'" in n or "[" in n for n in names)

    def test_js_export_assignments(self):
        r = _parse(JS_EXPORTS, language="javascript")
        assert _has(r, "mf", "function", None)
        assert _has(r, "ef", "function", None)
        assert _has(r, "named", "function", None)
        assert "exports" not in _names(r)
        assert GHOST not in _names(r)

    def test_rust_macro_rules(self):
        r = _parse("macro_rules! mymac { () => {} }\n", language="rust")
        assert _has(r, "mymac", "macro", None)
        assert GHOST not in _names(r)

    def test_java_annotation_element(self):
        r = _parse("@interface Ann { String v(); }\n", language="java")
        assert _has(r, "Ann", "interface", None)
        assert _has(r, "v", "method", "Ann")
        assert GHOST not in _names(r)

    def test_cpp_aliases_macros_and_function_pointers(self):
        r = _parse(CPP_EXTRA, language="cpp")
        assert _has(r, "Alias", "type", None)
        assert _has(r, "TD", "type", None)
        assert _has(r, "PTD", "type", None)
        assert _has(r, "Callback", "type", None)
        assert _has(r, "MAC", "macro", None)
        assert _has(r, "OBJ", "macro", None)
        assert _has(r, "getfp", "function", None)
        assert _has(r, "cb", "field", "S")
        assert not _has(r, "cb", "method", "S")
        assert _has(r, "m", "method", "S")
        assert _has(r, "f", "function", None)
        assert _has(r, "p", "function", None)
        assert _has(r, "q", "function", None)
        assert GHOST not in _names(r)
        assert not any("(" in n for n in _names(r))

    def test_cpp_object_construction_is_not_a_function(self):
        r = _parse(CPP_EXTRA, language="cpp")
        for name in ("w", "w2", "w3"):
            assert _has(r, name, "variable", None), name
            assert not _has(r, name, "function", None), name
        assert not {"config", "a", "b"} & _names(r)


CPP_MACRO_CLASS = """namespace ui::core {
class Config;
class WIDGET_API Widget : public Base {
public:
    explicit Widget(const Config& cfg);
    virtual ~Widget();
    virtual void render() const = 0;
    static Widget* create();
    int size() const { int local = 1; return local; }
private:
    int count_ = 0;
};
struct API S { void go(); };
class API Fin final { void fin(); };
}
"""

TS_WRAPPED = """const Btn = forwardRef((p, ref) => null);
export const UserRow = memo(function UserRow() { return null; });
const R = React.memo(withX((a) => a));
const h = asyncHandler(async (req, res) => {});
const Klass = mixin(class Inner { im() {} });
const f2 = ((x) => x) as Fn;
const f3 = (() => 1)!;
const f4 = (function () {}) satisfies Fn;
const cfg = { a: 1 } satisfies Cfg;
export const MAX = 10;
let counter = 0;
export const { left, right: renamed } = pair;
const fs = require('fs');
const helpers = { formatItem() {}, other: () => 1 };
function outer() { const localVar = 1; const localFn = () => localVar; }
"""

PY_ASSIGNMENTS = """import os
type ItemId = int
normalize = lambda s: s.strip()
MAX: int = 10
a, b = 1, 2
x = y = 0
os.environ["K"] = "v"
class C:
    norm = lambda self: 1
    attr = 3
def f():
    local = 1
"""

GO_VARS = """package main
var healthz = func(w int) {}
var Version = "1"
var (
    a, b = 1, func() {}
)
const (A = 1; B = 2)
func main() { var local = 1; _ = local }
"""


class TestWrappedAndValueDefinitions:
    """Definitions the second validator found reported absent."""

    def test_cpp_export_macro_class(self):
        r = _parse(CPP_MACRO_CLASS, language="cpp")
        assert _has(r, "Widget", "class", None)
        assert _has(r, "Widget", "method", "Widget")  # constructor
        assert _has(r, "~Widget", "method", "Widget")
        assert _has(r, "render", "method", "Widget")
        assert _has(r, "create", "method", "Widget")
        assert _has(r, "size", "method", "Widget")
        assert _has(r, "S", "struct", None)
        assert _has(r, "go", "method", "S")
        assert _has(r, "Fin", "class", None)
        assert _has(r, "fin", "method", "Fin")
        names = _names(r)
        assert GHOST not in names
        assert _has(r, "count_", "field", "Widget")
        assert {"WIDGET_API", "API", "final", "local", "Base"}.isdisjoint(names)
        assert not _has(r, "Widget", "function", None)

    def test_cpp_nested_namespace_segments(self):
        r = _parse(CPP_MACRO_CLASS, language="cpp")
        assert _has(r, "ui::core", "namespace", None)
        assert _has(r, "ui", "namespace", None)
        assert _has(r, "core", "namespace", None)
        assert GHOST not in _names(r)

    def test_ts_wrapped_and_unwrapped_functions(self):
        r = _parse(TS_WRAPPED, language="typescript")
        assert r["parse_error_line"] is None
        for name in ("Btn", "UserRow", "R", "h", "Klass", "f2", "f3", "f4"):
            assert _has(r, name, "const_function", None), name
        assert _has(r, "UserRow", "function", None)  # the named inner function
        assert _has(r, "Inner", "class", None)
        assert _has(r, "im", "method", "Inner")
        assert _has(r, "outer", "function", None)
        assert _has(r, "localFn", "const_function", None)
        names = _names(r)
        assert GHOST not in names
        assert {"forwardRef", "memo", "withX", "asyncHandler", "React"}.isdisjoint(names)

    def test_ts_top_level_variables(self):
        r = _parse(TS_WRAPPED, language="typescript")
        for name in ("cfg", "MAX", "counter", "left", "renamed", "helpers"):
            assert _has(r, name, "variable", None), name
        names = _names(r)
        assert GHOST not in names
        assert {"localVar", "fs", "right", "pair"}.isdisjoint(names)
        assert not _has(r, "cfg", "const_function", None)

    def test_js_object_method_parents_are_consistent(self):
        r = _parse(TS_WRAPPED, language="typescript")
        assert _has(r, "formatItem", "method", "helpers")
        assert _has(r, "other", "method", "helpers")
        assert not _has(r, "formatItem", "method", None)

    def test_python_module_assignments_type_alias_and_lambdas(self):
        r = _parse(PY_ASSIGNMENTS, language="python")
        assert r["parse_error_line"] is None
        assert _has(r, "ItemId", "type", None)
        assert _has(r, "normalize", "const_function", None)
        assert _has(r, "norm", "const_function", "C")
        for name in ("MAX", "a", "b", "x", "y"):
            assert _has(r, name, "variable", None), name
        names = _names(r)
        assert GHOST not in names
        assert _has(r, "attr", "field", "C")
        assert {"local", "environ", "int", "s", "self"}.isdisjoint(names)

    def test_go_top_level_vars_and_func_literals(self):
        r = _parse(GO_VARS, language="go")
        assert _has(r, "healthz", "const_function", None)
        assert _has(r, "b", "const_function", None)
        for name in ("Version", "a", "A", "B"):
            assert _has(r, name, "variable", None), name
        assert GHOST not in _names(r)
        assert "local" not in _names(r)

    def test_rust_trait_associated_type(self):
        r = _parse("trait T { type Error; fn f(&self); }\n", language="rust")
        assert _has(r, "Error", "type", "T")
        assert _has(r, "f", "method", "T")
        assert GHOST not in _names(r)

    def test_csharp_positional_record_and_namespace_segments(self):
        code = (
            "namespace Shop.Api.Controllers {\npublic record ProductDto(int Id, string Name);\n}\n"
        )
        r = _parse(code, language="c#")
        assert _has(r, "ProductDto", "class", None)
        assert _has(r, "Id", "property", "ProductDto")
        assert _has(r, "Name", "property", "ProductDto")
        for name in ("Shop.Api.Controllers", "Shop", "Api", "Controllers"):
            assert _has(r, name, "namespace", None), name
        names = _names(r)
        assert GHOST not in names
        assert {"int", "string"}.isdisjoint(names)

    def test_java_record_components_are_accessors(self):
        r = _parse("record P(int id, String name) { void m() {} }\n", language="java")
        assert _has(r, "P", "class", None)
        assert _has(r, "id", "method", "P")
        assert _has(r, "name", "method", "P")
        assert _has(r, "m", "method", "P")
        names = _names(r)
        assert GHOST not in names
        assert "String" not in names


RUST_CONSTS = """const MAX: u32 = 1;
static mut COUNTER: i32 = 0;
pub struct Foo;
impl Foo { const ID: u32 = 2; }
trait T { const TID: u32; fn f(&self); }
fn run() { const LOCAL: u32 = 1; let x = LOCAL; }
"""

CPP_VARIABLES = """namespace app {
int g_count = 0;
constexpr int kMax = 3;
extern int g_ext;
int *gp, arr[3];
int &gref = g_count;
template<class T> inline constexpr bool x_v = true;
auto lam = [](int a) { return a; };
}
int Store::total = 0;
template<class T, int N> struct RingBuffer {
    static constexpr size_type capacity = N;
    T *data_;
    static auto make = [](int n) { return n; };
};
template<class T> struct is_ring : std::false_type {};
template<class T, int N> struct is_ring<RingBuffer<T, N>> : std::true_type {};
template<> class Traits<int> {};
void work() { int loc = 1; auto inner = [](int q) { return q; }; consume(loc); }
"""

PY_FIELDS = """from django.db import models
class Post(models.Model):
    title = models.CharField(max_length=200)
    views: int = 0
    slug: str
    def save(self, *args):
        tmp = 1
        super().save(*args)
"""

TS_FIELDS = """class Store {
  count = 0;
  static label: string;
  #secret = 'x';
  handler = () => 1;
  sized = (1 as number);
}
app.post('/u', asyncHandler(async function createUser(req, res) {}));
function outer() { register(function innerNamed() {}); }
router.use((req, res) => next());
declare const dc: number;
declare let dl: string;
declare function df(): void;
declare class DC { m(): void; }
"""

JAVA_FIELDS = """class Account {
    private int balance, limit = 2;
    static final String KIND = "x";
    interface Codes { int OK = 0; }
    void run() { int local = 1; Object o = new Object() { int anon = 1; }; }
}
"""

CS_FIELDS = """class A {
    private int count, total;
    public const int Max = 1;
    ~A() {}
    void M() { int local = 1; }
}
struct P { public int X; }
"""

GO_FIELDS = """package p
type Server struct {
    Addr, Host string
    port int
    *log.Logger
    Base
}
func f() { var local int; _ = local }
"""


class TestFieldsAndScopedVariables:
    """Definitions the third validator found reported absent."""

    def test_rust_const_and_static(self):
        r = _parse(RUST_CONSTS, language="rust")
        assert _has(r, "MAX", "variable", None)
        assert _has(r, "COUNTER", "variable", None)
        assert _has(r, "ID", "field", "Foo")
        assert _has(r, "TID", "field", "T")
        names = _names(r)
        assert GHOST not in names
        assert {"LOCAL", "x", "u32"}.isdisjoint(names)

    def test_cpp_namespace_and_class_variables(self):
        r = _parse(CPP_VARIABLES, language="cpp")
        for name in ("g_count", "kMax", "g_ext", "gp", "arr", "gref", "x_v"):
            assert _has(r, name, "variable", None), name
        assert _has(r, "lam", "const_function", None)
        assert _has(r, "total", "field", "Store")
        assert _has(r, "capacity", "field", "RingBuffer")
        assert _has(r, "data_", "field", "RingBuffer")
        assert _has(r, "make", "const_function", "RingBuffer")
        names = _names(r)
        assert GHOST not in names
        assert {"loc", "inner", "q", "a", "n", "consume", "N", "T"}.isdisjoint(names)
        assert not any("*" in n or "[" in n or "&" in n for n in names)

    def test_cpp_template_specialisation_name(self):
        r = _parse(CPP_VARIABLES, language="cpp")
        assert _has(r, "is_ring", "struct", None)
        assert _has(r, "Traits", "class", None)
        assert not any("<" in n for n in _names(r))
        assert GHOST not in _names(r)

    def test_python_class_fields(self):
        r = _parse(PY_FIELDS, language="python")
        assert _has(r, "title", "field", "Post")
        assert _has(r, "views", "field", "Post")
        assert _has(r, "slug", "field", "Post")
        assert _has(r, "save", "method", "Post")
        names = _names(r)
        assert GHOST not in names
        assert {"tmp", "models", "CharField", "max_length", "super"}.isdisjoint(names)

    def test_ts_class_fields_named_function_args_and_declare(self):
        r = _parse(TS_FIELDS, language="typescript")
        assert r["parse_error_line"] is None
        for name in ("count", "label", "#secret", "sized"):
            assert _has(r, name, "field", "Store"), name
        assert _has(r, "handler", "method", "Store")
        assert not _has(r, "handler", "field", "Store")
        assert _has(r, "createUser", "function", None)
        assert _has(r, "innerNamed", "function", None)
        assert _has(r, "dc", "variable", None)
        assert _has(r, "dl", "variable", None)
        assert _has(r, "df", "function", None)
        assert _has(r, "DC", "class", None)
        assert _has(r, "m", "method", "DC")
        names = _names(r)
        assert GHOST not in names
        assert {"asyncHandler", "register", "app", "router", "next", "req", "res"}.isdisjoint(names)

    def test_js_named_function_expression_reported_once(self):
        code = "let expr = function named2() {};\nmodule.exports = function named() {};\n"
        r = _parse(code, language="javascript")
        assert [s["name"] for s in r["symbols"]].count("named2") == 1
        assert _has(r, "named2", "function", None)
        assert _has(r, "named", "function", None)
        assert _has(r, "expr", "const_function", None)
        assert GHOST not in _names(r)

    def test_java_fields(self):
        r = _parse(JAVA_FIELDS, language="java")
        for name in ("balance", "limit", "KIND"):
            assert _has(r, name, "field", "Account"), name
        assert _has(r, "OK", "field", "Codes")
        names = _names(r)
        assert GHOST not in names
        assert {"local", "o", "anon", "String"}.isdisjoint(names)

    def test_csharp_fields_and_destructor(self):
        r = _parse(CS_FIELDS, language="c#")
        for name in ("count", "total", "Max"):
            assert _has(r, name, "field", "A"), name
        assert _has(r, "X", "field", "P")
        assert _has(r, "~A", "method", "A")
        assert _has(r, "A", "class", None)
        names = _names(r)
        assert GHOST not in names
        assert "local" not in names
        assert not _has(r, "A", "method", "A")

    def test_go_struct_fields(self):
        r = _parse(GO_FIELDS, language="go")
        for name in ("Addr", "Host", "port"):
            assert _has(r, name, "field", "Server"), name
        names = _names(r)
        assert GHOST not in names
        # embedded types are defined elsewhere: never vouched for here
        assert {"local", "log", "Logger", "Base", "string", "int"}.isdisjoint(names)


def _sym_entry(result: dict, name: str, kind: str) -> dict:
    matches = [s for s in result["symbols"] if s["name"] == name and s["kind"] == kind]
    assert matches, f"no {kind} {name}: {result['symbols']}"
    return matches[0]


def _reexports(result: dict) -> list[tuple]:
    return [(e["name"], e["from_module"], e["alias"]) for e in result["reexports"]]


CPP_CONCEPTS = """template<typename T> concept Addable = requires(T a){ a + a; };
template<Addable T> T add(T a, T b);
template<typename T> requires Addable<T> void g(T);
namespace ns { class Base {}; }
class D : public virtual ns::Base, private Tmpl<int> {};
struct Plain {};
"""


class TestConceptsBasesAndReexports:
    """Facts that let a caller tell inherited / re-exported from absent."""

    def test_cpp_concept_defined_once(self):
        r = _parse(CPP_CONCEPTS, language="cpp")
        assert r["parse_error_line"] is None
        assert _has(r, "Addable", "concept", None)
        assert [s["name"] for s in r["symbols"]].count("Addable") == 1
        assert _has(r, "add", "function", None)
        assert _has(r, "g", "function", None)
        assert not any(s["kind"] == "concept" and s["name"] != "Addable" for s in r["symbols"])
        assert GHOST not in _names(r)

    def test_cpp_requires_clause_alone_defines_no_concept(self):
        code = "template<typename T> requires Addable<T> void g(T);\n"
        r = _parse(code, language="cpp")
        assert not any(s["kind"] == "concept" for s in r["symbols"])
        assert "Addable" not in _names(r)

    def test_cpp_bases(self):
        r = _parse(CPP_CONCEPTS, language="cpp")
        assert _sym_entry(r, "D", "class")["bases"] == ["ns::Base", "Tmpl"]
        assert "bases" not in _sym_entry(r, "Plain", "struct")
        assert "bases" not in _sym_entry(r, "Base", "class")
        assert {"Tmpl", "public", "virtual"}.isdisjoint(_names(r))
        assert r["reexports"] == [] and r["impls"] == []

    def test_python_bases(self):
        code = "class A(pkg.Base, Generic[T], metaclass=Meta):\n    pass\nclass B:\n    pass\n"
        r = _parse(code, language="python")
        assert _sym_entry(r, "A", "class")["bases"] == ["pkg.Base", "Generic"]
        assert "bases" not in _sym_entry(r, "B", "class")
        assert {"Meta", "metaclass", "Generic"}.isdisjoint(_names(r))

    def test_ts_bases(self):
        r = _parse(TYPESCRIPT, language="typescript")
        assert _sym_entry(r, "Circle", "class")["bases"] == ["Base", "Shape"]
        assert "bases" not in _sym_entry(r, "Shape", "interface")
        code = "interface I2 extends A, B.C<X> {}\nclass M extends ns.Base<T> {}\n"
        r = _parse(code, language="typescript")
        assert _sym_entry(r, "I2", "interface")["bases"] == ["A", "B.C"]
        assert _sym_entry(r, "M", "class")["bases"] == ["ns.Base"]

    def test_js_bases_skip_expressions(self):
        code = "class W extends ns.Base {}\nconst K = class extends Mixin(B) {};\nclass P {}\n"
        r = _parse(code, language="javascript")
        assert _sym_entry(r, "W", "class")["bases"] == ["ns.Base"]
        assert "bases" not in _sym_entry(r, "K", "class")
        assert "bases" not in _sym_entry(r, "P", "class")

    def test_java_bases(self):
        code = (
            "class A extends p.Base<T> implements I, q.J<X> {}\n"
            "interface K extends L, M<N> {}\n"
            "class Plain {}\n"
        )
        r = _parse(code, language="java")
        assert _sym_entry(r, "A", "class")["bases"] == ["p.Base", "I", "q.J"]
        assert _sym_entry(r, "K", "interface")["bases"] == ["L", "M"]
        assert "bases" not in _sym_entry(r, "Plain", "class")

    def test_csharp_bases(self):
        code = (
            "class A : Ns.Base<T>, IFoo {}\n"
            "record R(int X) : B(X), I;\n"
            "class Plain {}\n"
            "enum E : byte { One }\n"
        )
        r = _parse(code, language="c#")
        assert _sym_entry(r, "A", "class")["bases"] == ["Ns.Base", "IFoo"]
        assert _sym_entry(r, "R", "class")["bases"] == ["B", "I"]
        assert "bases" not in _sym_entry(r, "Plain", "class")
        assert "bases" not in _sym_entry(r, "E", "enum")

    def test_go_embedded_types_are_bases_not_symbols(self):
        r = _parse(GO_FIELDS, language="go")
        assert _sym_entry(r, "Server", "struct")["bases"] == ["log.Logger", "Base"]
        assert {"log", "Logger", "Base", "log.Logger"}.isdisjoint(_names(r))
        code = "package p\ntype I interface {\n  io.Reader\n  M()\n}\ntype J interface { M() }\n"
        r = _parse(code, language="go")
        assert _sym_entry(r, "I", "interface")["bases"] == ["io.Reader"]
        assert "bases" not in _sym_entry(r, "J", "interface")
        assert "bases" not in _sym_entry(_parse(GO, language="go"), "Rect", "struct")

    def test_rust_impls(self):
        r = _parse(RUST, language="rust")
        assert r["impls"] == [{"type": "Counter", "trait": "Tick", "line": 21}]
        assert "bases" not in _sym_entry(r, "Counter", "struct")
        code = "impl<T> fmt::Display for Wrapper<T> {}\nimpl !Send for Raw {}\nimpl Raw {}\n"
        r = _parse(code, language="rust")
        assert [(i["type"], i["trait"]) for i in r["impls"]] == [("Wrapper", "fmt::Display")]

    def test_js_reexports(self):
        code = (
            'import { x, y as w } from "./p";\n'
            'export { a, b as c } from "./m";\n'
            'export * from "./n";\n'
            'export * as ns from "./o";\n'
            "export { x, w as q, localOnly };\n"
            "export const k = 1;\n"
            "export default k;\n"
            "const localOnly = 2;\n"
        )
        r = _parse(code, language="typescript")
        assert _reexports(r) == [
            ("a", "./m", None),
            ("b", "./m", "c"),
            ("*", "./n", None),
            ("*", "./o", "ns"),
            ("x", "./p", None),
            ("w", "./p", "q"),
            ("localOnly", None, None),
        ]
        assert r["reexports"][0]["line"] == 2
        assert not any(e["name"] == "k" for e in r["reexports"])

    def test_plain_export_is_not_a_reexport(self):
        r = _parse("export const k = 1;\nexport function f() {}\n", language="javascript")
        assert r["reexports"] == []

    def test_python_init_reexports(self):
        code = "from .m import x, y as z\nfrom pkg.sub import *\nimport os\n"
        r = _parse(code, language="python", file_path="pkg/__init__.py")
        assert _reexports(r) == [
            ("x", ".m", None),
            ("y", ".m", "z"),
            ("*", "pkg.sub", None),
        ]

    def test_python_non_init_has_no_reexports(self):
        code = "from .m import x\n"
        assert _parse(code, language="python", file_path="pkg/mod.py")["reexports"] == []
        assert _parse(code, language="python")["reexports"] == []

    def test_rust_reexports(self):
        code = (
            "pub use a::b::C;\n"
            "pub use a::b::{D, E as F};\n"
            "pub use a::*;\n"
            "use x::Y;\n"
            "pub(crate) use z::W;\n"
        )
        r = _parse(code, language="rust")
        assert _reexports(r) == [
            ("C", "a::b", None),
            ("D", "a::b", None),
            ("E", "a::b", "F"),
            ("*", "a", None),
        ]

    def test_other_languages_have_empty_file_level_lists(self):
        r = _parse(JAVA, language="java")
        assert r["reexports"] == [] and r["impls"] == []


class TestSyntaxErrorAndResolution:
    def test_syntax_error_keeps_earlier_symbols(self):
        code = "def good():\n    return 1\n\nclass Fine:\n    pass\n\ndef broken(:\n"
        r = _parse(code, language="python")
        assert _has(r, "good", "function")
        assert _has(r, "Fine", "class")
        assert r["parse_error_line"] == 7

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="supported languages"):
            extract_symbols(b"x", language="cobol")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="supported languages"):
            extract_symbols(b"x", file_path="a.rb")

    def test_no_language_raises(self):
        with pytest.raises(ValueError):
            extract_symbols(b"x")

    def test_missing_grammar_is_not_parsed(self, monkeypatch):
        monkeypatch.setattr(hunt, "get_parser", lambda grammar: None)
        r = extract_symbols(b"def f(): pass\n", language="python")
        assert r["parsed"] is False
        assert r["symbols"] == []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _make_mock_user(is_admin=False):
    user = MagicMock()
    user.preferred_username = "test-user"
    user.is_admin = MagicMock(return_value=is_admin)
    user.effective_tenant_id = "test-tenant"
    return user


class TestSymbolsEndpoint:
    @pytest.mark.asyncio
    async def test_200_shape(self):
        from app.api.endpoints.analysis_modules.symbols import (
            SymbolsRequest,
            extract_file_symbols,
        )

        response = await extract_file_symbols(
            request=SymbolsRequest(code="def f():\n    pass\n", file_path="a.py"),
            user=_make_mock_user(),
        )
        assert response.language == "python"
        assert response.parsed is True
        assert [(s.name, s.kind, s.parent) for s in response.symbols] == [("f", "function", None)]

    @pytest.mark.asyncio
    async def test_422_unsupported_language(self):
        from app.api.endpoints.analysis_modules.symbols import (
            SymbolsRequest,
            extract_file_symbols,
        )

        with pytest.raises(HTTPException) as exc_info:
            await extract_file_symbols(
                request=SymbolsRequest(code="x", language="cobol"), user=_make_mock_user()
            )
        assert exc_info.value.status_code == 422
        assert "python" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_422_no_language_or_path(self):
        from app.api.endpoints.analysis_modules.symbols import (
            SymbolsRequest,
            extract_file_symbols,
        )

        with pytest.raises(HTTPException) as exc_info:
            await extract_file_symbols(request=SymbolsRequest(code="x"), user=_make_mock_user())
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_503_when_grammar_unavailable(self, monkeypatch):
        from app.api.endpoints.analysis_modules.symbols import (
            SymbolsRequest,
            extract_file_symbols,
        )

        monkeypatch.setattr(hunt, "get_parser", lambda grammar: None)
        with pytest.raises(HTTPException) as exc_info:
            await extract_file_symbols(
                request=SymbolsRequest(code="def f(): pass\n", language="python"),
                user=_make_mock_user(),
            )
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "grammar not available: python"

    def test_route_and_operation_id(self):
        from app.api.endpoints.analysis_modules.symbols import router
        from auth.tenant import get_current_user_with_tenant

        app = FastAPI()
        app.include_router(router, prefix="/api/analysis")
        app.dependency_overrides[get_current_user_with_tenant] = lambda: _make_mock_user()
        client = TestClient(app)

        ok = client.post(
            "/api/analysis/symbols",
            json={"code": "export const f = () => 1;\n", "file_path": "a.ts"},
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["symbols"][0]["kind"] == "const_function"

        bad = client.post("/api/analysis/symbols", json={"code": "x", "language": "cobol"})
        assert bad.status_code == 422

        operation = app.openapi()["paths"]["/api/analysis/symbols"]["post"]
        assert operation["operationId"] == "extract_symbols"
