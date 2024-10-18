#!/usr/bin/env python3
import ast
import inspect
import logging
import os
import re
import subprocess
import tempfile
import types
import zipfile
from functools import partial
from typing import List, TextIO

import astor
import torch
import triton

from .reporting import ErrorAggregatorDict, Stats
from .static_analysis import (
    ASTCleanup,
    CheckCallableMembers,
    CONFIG_NAMES,
    ExtractConfigUsage,
    ExtractReadsWrites,
    IMPORT_WHITELIST,
    split_import,
)
from .utils import call_with_timeout

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(\btorch\b)|(\bnn[.]Module\b)", re.MULTILINE)
PREFIX = f"""
from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import {', '.join(sorted(IMPORT_WHITELIST))}
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor

patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
{" = ".join(sorted(CONFIG_NAMES))} = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = "1.0.0"
xrange = range
wraps = functools.wraps
"""
SUFFIX = """
import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

"""


def to_source(node):
    return astor.to_source(
        node,
        pretty_source="".join,
        pretty_string=partial(astor.string_repr.pretty_string, max_line=8192),
    )


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(
        self,
        tempdir: str,
        errors: ErrorAggregatorDict,
        stats: Stats,
        output_py: TextIO,
        args,
    ):
        super(PyTorchModuleExtractor, self).__init__()
        self.errors = errors
        self.stats = stats

        self.output = IncrementalModule(tempdir, output_py)

        self.imports = dict()
        self.constants = []

        self.available_symbols = dict()
        self.global_config = None

        self.testcases = []
        self.args = args
        self.triton_kernel_names = []  # list of triton kernels in the input project

    def search_file(self, filename: str, open_fn=open):
        """get module from filename .py file"""
        if not filename.endswith(".py") or ".#" in filename:
            return

        with open_fn(filename, "r") as fp:
            source = fp.read()
            if isinstance(source, bytes):
                source = source.decode("utf-8")

        has_match = bool(NN_MODULE_RE.search(source))  # there is torch in .py

        try:
            tree = self.ast_parse(source, filename)
        except Exception as e:
            return self.errors.record("parse", e)

        m = re.search(r"([a-z0-9_]+)/__init__.py$", filename, re.I)
        if m:
            self.output.add_module_alias(m.group(1), has_match)
        else:
            self.output.add_module_alias(
                os.path.splitext(os.path.basename(filename))[0], has_match
            )

        self.search_ast(tree, has_match)

    @staticmethod
    def ast_parse(source, filename):
        try:
            return ast.parse(source, filename)  # get ast nodes from code
        except SyntaxError:
            # perhaps python2?
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".py") as tmp:
                tmp.write(
                    re.sub(r"\basync *=", "non_blocking=", source)
                    .replace("\t", "    ")
                    .encode("utf-8")
                )
                tmp.flush()
                with open("/dev/null", "w") as null:
                    subprocess.check_call(
                        ["2to3", "-w", tmp.name], stderr=null, stdout=null
                    )
                return ast.parse(open(tmp.name).read(), filename)

    def has_triton_decorator(self, node: ast.AST) -> bool:
        """Check if a node is a triton.jit decorator"""
        if isinstance(node, ast.Name) and node.id == "jit":
            return True
        elif isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name)
                and node.value.id == "triton"
                and node.attr == "jit"
            )
        return False

    def search_ast(self, tree: ast.AST, overwrite: bool):
        """get torch classes, import and functions"""
        scope = types.ModuleType("_scope")
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self.add_available_symbol(node, overwrite)
                bases = [to_source(x).strip() for x in node.bases]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if overwrite:
                    for module_name, import_node in split_import(node):
                        if module_name in IMPORT_WHITELIST:
                            self.imports[to_source(import_node)] = import_node
            elif isinstance(node, ast.FunctionDef) and any(
                self.has_triton_decorator(d) for d in node.decorator_list
            ):
                self.add_available_symbol(node, overwrite)
                self.triton_kernel_names.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)):
                self.add_available_symbol(node, overwrite)

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def search_zipfile(self, filename: str):
        with zipfile.ZipFile(filename) as archive:
            for name in sorted(archive.namelist()):
                self.search_file(name, archive.open)

    def add_available_symbol(self, node, overwrite=False):
        node = ast.fix_missing_locations(ASTCleanup().visit(node))  # clean ast
        try:
            if overwrite:
                self.available_symbols[node.name] = node
            else:
                self.available_symbols.setdefault(node.name, node)
        except AttributeError:  # node.name is missing
            reads, writes = ExtractReadsWrites.run(node)
            for name in writes:
                if overwrite:
                    self.available_symbols[name] = node
                else:
                    self.available_symbols.setdefault(name, node)

    def construct_module(self):
        self.output.run_statement(
            self.ast_parse(PREFIX, "<string>"), source_required=True
        )
        self.global_config = self.output.output_module.__dict__["_global_config"]
        self.name_to_ast = dict()

        for statement in self.imports.values():
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("import", e, "")
        for statement in self.constants:
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("constant", e, getattr(statement, "name", ""))
        for name in self.triton_kernel_names:
            statement = self.available_symbols.get(name)
            if statement:
                self.add_requirements(statement)  # add what is needed for module
                try:
                    self.run_statement(statement)
                    self.available_symbols.pop(name)
                except Exception as e:
                    self.errors.record("define", e, getattr(statement, "name", ""))

    def run_statement(self, statement):
        self.output.run_statement(statement, source_required=True)
        name = getattr(statement, "name", None)
        if name:
            self.name_to_ast[name] = statement

    def add_requirements(self, statement):
        """
        Recursively add symbols to the output module needed by statement.

        :param statement: ast.Node to add to the module
        """
        reads, writes = ExtractReadsWrites.run(statement)
        needs = {sym for sym in reads - writes if sym not in self.output}
        log.debug(
            f"add_requirements: {getattr(statement, 'name', '')} "
            f"available {needs & set(self.available_symbols.keys())} "
            f"unavailable {needs - set(self.available_symbols.keys())}"
        )

        need_config = False
        for name in sorted(needs):
            if name in self.available_symbols and name not in self.output:
                requirement = self.available_symbols.get(name)
                self.add_requirements(requirement)
                try:
                    self.run_statement(requirement)
                    self.available_symbols.pop(name)
                except:
                    log.warning("Error adding requirement", exc_info=True)
            elif name in CONFIG_NAMES:
                need_config = True

        if need_config:
            try:
                for key in ExtractConfigUsage.run(statement):
                    if key not in sorted(self.global_config):
                        value = repr(DeduceParameter.initial_arg_init(key, None))
                        self.output.run_statement(
                            self.ast_parse(
                                f"_global_config['{key}'] = {value}\n", "<string>"
                            ),
                            source_required=True,
                        )
            except Exception:
                log.exception("global_config error")

    def main(self, filename: str):
        basename = re.sub(r"[.]zip$", "", os.path.basename(filename))

        self.output.writelines(
            ["import sys\n", "_module = sys.modules[__name__]\n", "del sys\n"]
        )

        if os.path.isdir(filename):
            self.search_directory(filename)  # find nn modules, imports, symbols
        else:
            self.search_zipfile(filename)

        self.construct_module()  # run and write ast nodes

        log.info(f"{basename}: {self.stats}")


class IncrementalModule(object):
    """
    Construct a python module statement by statement, recording the result
    to a generated python file.
    """

    def __init__(self, tempdir: str, output_py: TextIO):
        super().__init__()
        self.tempdir = tempdir
        self.output_module = types.ModuleType(f"{__name__}.output")
        self.output_py = output_py

    def __contains__(self, name):
        """
        :param name: symbol to check for
        :return: True if output module contains name (and it is not an alias)
        """
        return (
            getattr(self.output_module, name, self.output_module)
            is not self.output_module
        )

    def items(self):
        return self.output_module.__dict__.items()

    def same_module(self, obj):
        """
        :param obj: a python object
        :return: True if obj is defined in this module
        """
        return obj.__module__ == self.output_module.__name__

    def write(self, data: str):
        self.output_py.write(data)

    def writelines(self, data: List[str]):
        self.output_py.writelines(data)

    def run_statement(self, statement, source_required=False):
        """
        Runs a ast statement node and writes code into output_py
        """
        source = to_source(statement)
        if not source_required:
            code = compile(ast.Module([statement], []), "<string>", "exec")
        else:
            # TorchScript requires source code to exist on disk
            assert self.tempdir
            fn, filename = tempfile.mkstemp(suffix=".py", dir=self.tempdir, prefix="pb")
            with os.fdopen(fn, "w") as fd:
                fd.write(source)
                fd.flush()
            code = compile(source, filename, "exec")
        exec(code, self.output_module.__dict__, self.output_module.__dict__)
        self.output_py.writelines(["\n", source, "\n"])

    def add_module_alias(self, name: str, overwrite: bool):
        """
        We flatten everything we extract into a single module, this adds
        a symbol to that unified module that points to the same module
        so that internal a.b.c references work.

        :param name: alternate name for self.output_module
        :param overwrite: if true, replace an existing symbol
        """
        if name in {
            "global",
            "try",
            "except",
            "if",
            "in",
            "else",
            "for",
            "import",
            "pass",
            "return",
            "def",
            "int",
            "super",
            "torch",
            "__main__",
        }:
            return
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return
        if name in self.output_module.__dict__ and not overwrite:
            return
        self.output_module.__dict__[name] = self.output_module
        self.output_py.write(f"{name} = _module\n")
