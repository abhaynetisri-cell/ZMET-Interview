import pytest
import os
from unittest.mock import MagicMock

# Import modules to test
from llm_client import ZMETLLMClient
from interview_engine import (
    ZMETSession,
    ZMETInterviewEngine,
    Construct,
    Relation,
    ImageMetadata
)

def test_session_schema_defaults():
    """Verify that a newly instantiated ZMETSession has the correct defaults."""
    session = ZMETSession(
        session_id="test_sess_01",
        started_at="2026-07-15T18:00:00Z",
        respondent_id="resp_abc"
    )
    assert session.session_id == "test_sess_01"
    assert session.current_step == 3
    assert len(session.images) == 0
    assert len(session.constructs) == 0
    assert len(session.relations) == 0
    assert len(session.transcript) == 0

def test_engine_state_transitions():
    """Test state engine helpers and transition structures."""
    # Mock LLM client so we don't hit live APIs during tests
    mock_llm = MagicMock(spec=ZMETLLMClient)
    engine = ZMETInterviewEngine(llm_client=mock_llm)
    
    # 1. Test construct ID sequence generation
    c1 = engine.get_next_construct_id()
    assert c1 == "c_01"
    
    # Add construct to database
    engine.session.constructs.append(Construct(id=c1, name="Test Construct", type="attribute", source_step=3))
    
    c2 = engine.get_next_construct_id()
    assert c2 == "c_02"
    
    # 2. Test relation ID sequence generation
    r1 = engine.get_next_relation_id()
    assert r1 == "r_01"

def test_triad_selection():
    """Verify that triad selection behaves correctly depending on image count."""
    mock_llm = MagicMock(spec=ZMETLLMClient)
    engine = ZMETInterviewEngine(llm_client=mock_llm)
    
    # Empty images - should return None
    assert engine.select_triad() is None
    
    # 2 images - should return None
    engine.session.images.append(ImageMetadata(image_id="img_01", filename="a.jpg", user_narration="A"))
    engine.session.images.append(ImageMetadata(image_id="img_02", filename="b.jpg", user_narration="B"))
    assert engine.select_triad() is None
    
    # 3 images - should return the list of 3 images
    engine.session.images.append(ImageMetadata(image_id="img_03", filename="c.jpg", user_narration="C"))
    triad = engine.select_triad()
    assert triad is not None
    assert len(triad) == 3
    assert triad[0].image_id == "img_01"
    assert triad[2].image_id == "img_03"

def test_pdf_generation():
    """Verify that a PDF can be successfully generated from a ZMET session."""
    from pdf_generator import generate_zmet_pdf
    
    # Create mock session
    session = ZMETSession(
        session_id="test_sess_02",
        started_at="2026-07-15T18:00:00Z",
        respondent_id="resp_xyz"
    )
    # Populate mock data
    session.constructs.append(Construct(id="c_01", name="Quiet space", type="attribute", source_step=3))
    session.constructs.append(Construct(id="c_02", name="Relaxation", type="consequence", source_step=6))
    session.relations.append(Relation(relation_id="r_01", source="c_01", target="c_02", explanation="Quiet leads to relaxation"))
    session.sensory_analogies.taste = "Clean water"
    session.vignette.summary_story = "This is a closing vignette summary story."
    session.vignette.researcher_summary = "This is a researcher summary."
    session.transcript.append({"sender": "moderator", "text": "Hello"})
    session.transcript.append({"sender": "respondent", "text": "Hi"})
    
    # Generate PDF bytes
    pdf_bytes = generate_zmet_pdf(session)
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # PDF header signature Check
    assert pdf_bytes.startswith(b"%PDF")

