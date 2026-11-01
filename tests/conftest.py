"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _offline_token_counting():
    """Inspect counts tokens with tiktoken, which downloads its encoding on first use.

    In sandboxes without that network access, fall back to a rough estimate so the
    end-to-end tests still run. Where the download works, nothing is patched.
    """
    try:
        import tiktoken

        tiktoken.get_encoding("o200k_base")
        yield
        return
    except Exception:
        pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "inspect_ai.model._model.count_text_tokens", lambda text: max(1, len(text) // 4)
        )
        yield
