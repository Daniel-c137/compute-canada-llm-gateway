import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    from cc_llm_gateway.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
