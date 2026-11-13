"""Drop-in orchestration framework integrations for Conflux.

Adapters work out of the box with zero third-party dependencies, and
automatically inherit each framework's native base classes / protocols when
``langgraph``, ``crewai``, or ``autogen-core`` is installed:

- **LangGraph**: :class:`ConfluxCheckpointSaver`, :class:`ConfluxStore`
- **CrewAI**: :class:`ConfluxCrewStorage`, :class:`ConfluxFlowPersistence`,
  :func:`create_crewai_tools`
- **AutoGen**: :class:`ConfluxAutoGenMemory`, :func:`create_autogen_tools`
"""

from .autogen import ConfluxAutoGenMemory, create_autogen_tools
from .base import BackendAdapter, coerce_backend
from .crewai import (
    ConfluxCallableTool,
    ConfluxCrewStorage,
    ConfluxFlowPersistence,
    FallbackMemoryRecord,
    FallbackScopeInfo,
    create_crewai_tools,
)
from .langgraph import ConfluxCheckpointSaver, ConfluxStore

__all__ = [
    "BackendAdapter",
    "coerce_backend",
    "ConfluxCheckpointSaver",
    "ConfluxStore",
    "ConfluxCrewStorage",
    "ConfluxFlowPersistence",
    "ConfluxCallableTool",
    "FallbackMemoryRecord",
    "FallbackScopeInfo",
    "create_crewai_tools",
    "ConfluxAutoGenMemory",
    "create_autogen_tools",
]
