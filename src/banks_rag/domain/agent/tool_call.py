"""Value object ``ToolCall`` — una llamada a herramienta parseada del LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Una invocación a herramienta extraída del output del LLM.

    Campos:
        id: identificador único (uuid12) para correlacionar la respuesta tool→assistant.
        name: nombre del tool a invocar (debe estar registrado).
        arguments: dict de argumentos parseado del JSON emitido por el modelo.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_message_block(self) -> dict:
        """Bloque OpenAI-compat para insertar como ``assistant`` message.

        El siguiente paso del loop espera el resultado de la tool con
        ``role='tool'`` y ``tool_call_id`` igual al ``id`` de este block.
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
