"""
Auxiliares de execução: leitura de variável de ambiente e silêncio nos avisos
barulhentos das dependências.
"""

from __future__ import annotations

import logging
import os
import warnings

# Valores de exemplo do .env.template. Se a variável ainda tem um deles, a
# chave de verdade não foi colocada.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def has_env(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip() and value.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    """
    Cala avisos que as dependências soltam ao importar. Tem que rodar *antes*
    de importar o forecasting_tools.
    """
    warnings.filterwarnings("ignore", message=r".*does not support cost tracking.*")
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # O aviso de token ausente sai na importação; o main.py confere as chaves por conta própria.
    logging.getLogger("forecasting_tools.helpers.metaculus_client").setLevel(
        logging.ERROR
    )
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False
