"""
Detect-Labeler Core Package
===========================
Agnostic annotation converters, smart CV snapping, and model integration hooks.
"""

from core.converter import UniversalConverter, AnnotationBox
from core.smart_annotator import SmartAnnotator
from core.model_adapter import ModelAdapter

__all__ = ["UniversalConverter", "AnnotationBox", "SmartAnnotator", "ModelAdapter"]
