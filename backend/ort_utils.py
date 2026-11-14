"""Shared ONNX Runtime session settings.

Every InferenceSession in this backend is built through here. Two sessions
created with default options each spawn a full spin-waiting thread pool and
starve one another -- measured at ~2.3 s per face frame. Capping the pool and
turning off spinning keeps every model fast when they run back to back.
"""

import onnxruntime as ort

import config

_SILENCED = False


def session_options() -> "ort.SessionOptions":
    global _SILENCED
    if not _SILENCED:
        ort.set_default_logger_severity(3)   # drop per-run shape warnings
        _SILENCED = True
    so = ort.SessionOptions()
    so.intra_op_num_threads = config.ORT_THREADS
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def make_session(model_path: str) -> "ort.InferenceSession":
    return ort.InferenceSession(str(model_path), session_options(),
                                providers=["CPUExecutionProvider"])
