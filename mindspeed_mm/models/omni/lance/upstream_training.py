"""Audited execution policy for the released Lance training entrypoint.

The released training loop logs broad step exceptions and then continues.  A
launcher cannot distinguish such a run from a successful run when every batch
fails.  This module performs one narrow, in-memory AST transformation: the
known training-step handler keeps its diagnostics and gradient cleanup, drops
the handler-only barrier, and re-raises the original exception.  The upstream
checkout is never modified.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import CodeType
from typing import Any, Dict, List, Tuple, Union


class LanceUpstreamTrainingError(RuntimeError):
    """Raised when the upstream entrypoint cannot enforce fail-fast training."""


_TRAINING_EXCEPTION_MARKER = "[TRAINING EXCEPTION]"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contains_training_marker(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and _TRAINING_EXCEPTION_MARKER in child.value
        for child in ast.walk(node)
    )


def _contains_barrier_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "barrier"
        for child in ast.walk(node)
    )


def _is_broad_exception(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or (
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
    )


class _FailFastStepTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.handlers: List[Dict[str, Any]] = []

    def visit_Try(self, node: ast.Try) -> ast.AST:
        self.generic_visit(node)
        for handler in node.handlers:
            if not _is_broad_exception(handler) or not _contains_training_marker(handler):
                continue
            continue_statements = [
                statement for statement in handler.body if isinstance(statement, ast.Continue)
            ]
            barrier_statements = [
                statement for statement in handler.body if _contains_barrier_call(statement)
            ]
            self.handlers.append(
                {
                    "line": handler.lineno,
                    "continue_statements": len(continue_statements),
                    "barrier_statements": len(barrier_statements),
                }
            )
            if len(continue_statements) != 1 or len(barrier_statements) != 1:
                continue
            handler.body = [
                statement
                for statement in handler.body
                if not isinstance(statement, ast.Continue)
                and not _contains_barrier_call(statement)
            ]
            handler.body.append(ast.copy_location(ast.Raise(exc=None, cause=None), handler))
        return node


def _transform_entrypoint(path: Path) -> Tuple[ast.Module, Dict[str, Any]]:
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise LanceUpstreamTrainingError(
            "could not read upstream training entrypoint {}: {}".format(path, exc)
        ) from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise LanceUpstreamTrainingError(
            "could not parse upstream training entrypoint {}: {}".format(path, exc)
        ) from exc

    transformer = _FailFastStepTransformer()
    transformed = transformer.visit(tree)
    ast.fix_missing_locations(transformed)
    valid_handlers = [
        item
        for item in transformer.handlers
        if item["continue_statements"] == 1 and item["barrier_statements"] == 1
    ]
    issues = []
    if len(transformer.handlers) != 1:
        issues.append(
            "expected exactly one released Lance training-step exception handler, found {}".format(
                len(transformer.handlers)
            )
        )
    elif len(valid_handlers) != 1:
        issues.append("the released Lance training-step exception handler has changed shape")
    metadata = {
        "status": "valid" if not issues else "invalid",
        "policy": "fail-fast-on-training-step-exception",
        "entrypoint": str(path),
        "source_sha256": _sha256_bytes(source),
        "upstream_file_modified": False,
        "handlers": transformer.handlers,
        "issues": issues,
    }
    return transformed, metadata


def validate_strict_training_entrypoint(
    entrypoint: Union[str, Path],
) -> Dict[str, Any]:
    """Confirm that the narrow fail-fast transformation matches upstream."""

    path = Path(entrypoint).expanduser().resolve()
    try:
        _, metadata = _transform_entrypoint(path)
    except LanceUpstreamTrainingError as exc:
        return {
            "status": "invalid",
            "policy": "fail-fast-on-training-step-exception",
            "entrypoint": str(path),
            "upstream_file_modified": False,
            "handlers": [],
            "issues": [str(exc)],
        }
    return metadata


def compile_strict_training_entrypoint(
    entrypoint: Union[str, Path],
) -> Tuple[CodeType, Dict[str, Any]]:
    """Compile the released entrypoint with the audited fail-fast policy."""

    path = Path(entrypoint).expanduser().resolve()
    tree, metadata = _transform_entrypoint(path)
    if metadata["status"] != "valid":
        raise LanceUpstreamTrainingError("; ".join(metadata["issues"]))
    return compile(tree, str(path), "exec"), metadata
