"""Test VLM prompt construction with financial document instructions."""
import pytest
from src.services.vlm_service import _build_messages, _FIN_DOC_SYSTEM
from src.models.schema import PageResult


def test_build_messages_includes_system_instruction():
    """Verify financial document analysis instruction is prepended."""
    pages = [
        PageResult(report_id="AAPL_2023", page_num=1, image_base64="fake_b64", maxsim_score=0.9)
    ]
    messages = _build_messages("What is revenue?", pages)
    
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    
    # First block should be system instruction
    assert content[0]["type"] == "text"
    assert _FIN_DOC_SYSTEM in content[0]["text"]
    
    # Second block should be user query
    assert content[1]["type"] == "text"
    assert "What is revenue?" in content[1]["text"]
    
    # Third block should be image
    assert content[2]["type"] == "image_url"


def test_system_instruction_covers_failure_modes():
    """Ensure system instruction addresses known failure modes."""
    # Mode 1: Cover/TOC page filtering
    assert "IGNORE cover pages" in _FIN_DOC_SYSTEM
    assert "table-of-contents" in _FIN_DOC_SYSTEM
    
    # Mode 2: Evidence insufficiency handling
    assert "Insufficient evidence" in _FIN_DOC_SYSTEM
    assert "do not contain sufficient evidence" in _FIN_DOC_SYSTEM
    
    # Mode 3: Priority guidance
    assert "Prioritize" in _FIN_DOC_SYSTEM
    assert "tables" in _FIN_DOC_SYSTEM.lower()
    assert "Management Discussion" in _FIN_DOC_SYSTEM


def test_prompt_length_is_moderate():
    """Verify system instruction is concise (< 100 words to avoid token waste)."""
    word_count = len(_FIN_DOC_SYSTEM.split())
    assert word_count < 100, f"System instruction too long: {word_count} words"
