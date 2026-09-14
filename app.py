import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Import local modules
from llm_client import ZMETLLMClient
from interview_engine import (
    ZMETSession,
    ZMETInterviewEngine,
    ImageMetadata,
    Construct,
    Relation
)
from graph_renderer import generate_pyvis_html
from pdf_generator import generate_zmet_pdf

# Set page config
st.set_page_config(
    page_title="ZMET AI Moderator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling -- forced dark theme via !important overrides on
# Streamlit's actual container elements. This does not depend on
# .streamlit/config.toml or the Settings-menu theme choice at all, so it
# can't be silently overridden by either of those.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force dark background on every top-level Streamlit container */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    body {
        background-color: #0E1117 !important;
        color: #F4F5FA !important;
    }

    /* Force dark styling on every native input widget, using broad
       selectors that target the raw HTML elements directly -- testid-based
       selectors weren't matching reliably across Streamlit versions. */
    input, textarea, select,
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #1C1E29 !important;
        color: #F4F5FA !important;
        border-color: #333648 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #6B6685 !important;
    }
    li[role="option"]:hover {
        background-color: #2A2C46 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #1C1E29 !important;
        border-color: #333648 !important;
    }

    /* Headings and body text -- force light text everywhere */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #F4F5FA;
    }

    .zmet-card {
        background-color: #1C1E29 !important;
        color: #F4F5FA !important;
        padding: 28px 32px;
        border-radius: 14px;
        border: 1px solid #333648;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }

    .zmet-header {
        font-size: 22px;
        font-weight: 600;
        color: #F4F5FA !important;
        margin-bottom: 6px;
    }

    .zmet-subheader {
        font-size: 14px;
        color: #9B9BB5 !important;
        margin-bottom: 22px;
    }

    /* Thin, subtle divider -- replaces the old bare "---" horizontal bars */
    .zmet-divider {
        border: none;
        border-top: 1px solid #333648;
        margin: 20px 0;
    }

    /* Compact top-of-page step tracker (replaces the old sidebar list) */
    .step-tracker {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 22px;
    }
    .step-pill {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 500;
        background-color: #1C1E29;
        color: #9B9BB5 !important;
        border: 1px solid #333648;
    }
    .step-pill-active {
        background-color: #818CF8;
        color: #12131C !important;
        border: 1px solid #818CF8;
    }
    .step-pill-done {
        background-color: #2A2C46;
        color: #C7D2FE !important;
        border: 1px solid #4C4F73;
    }

    /* Chat bubbles in Step 6 */
    .chat-moderator {
        background-color: #23263A !important;
        color: #F4F5FA !important;
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #333648;
    }
    .chat-respondent {
        background-color: #2A2C46 !important;
        color: #D6D9FF !important;
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #4C4F73;
    }

    /* Buttons -- brighter accent that pops against dark */
    .stButton > button {
        background-color: #818CF8 !important;
        color: #12131C !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
    .stButton > button:disabled {
        background-color: #333648 !important;
        color: #6B6685 !important;
    }

    /* Tooltip styling helper */
    .info-icon {
        cursor: pointer;
        color: #818CF8;
        display: inline-block;
        margin-left: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def divider():
    """A subtle 1px divider instead of Streamlit's bare st.markdown('---') bar."""
    st.markdown('<hr class="zmet-divider"/>', unsafe_allow_html=True)

# =====================================================================
# 1. State Management Initialize
# =====================================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

if "model_name" not in st.session_state:
    # The entire gemini-2.5-* generation is restricted for new API
    # keys/projects. gemini-flash-latest is a Google-maintained alias,
    # confirmed working via debug_step6.py, that avoids re-chasing pinned
    # version deprecations going forward.
    st.session_state.model_name = "gemini-flash-latest"

if "engine" not in st.session_state:
    st.session_state.engine = None

if "storytelling_data" not in st.session_state:
    st.session_state.storytelling_data = {}  # image_idx: narration

# =====================================================================
# 2. Sidebar Navigation & Global Config
# =====================================================================

with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Input your Gemini API Key. Get one for free from Google AI Studio."
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if st.session_state.engine:
            st.session_state.engine.llm.set_api_key(api_key_input)
            
    # Model Selection
    # NOTE: gemini-2.0-flash and the entire gemini-2.5-* generation are
    # shut down/restricted for new API keys as of mid-2026.
    # gemini-flash-latest (a Google-maintained alias) is confirmed working
    # via debug_step6.py and is listed first.
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="gemini-flash-latest is confirmed working on this account. The others are listed as fallbacks if you hit rate limits or Google changes availability again -- run debug_step6.py with ZMET_TEST_MODEL set to test any of them first."
    )
    if model_choice != st.session_state.model_name:
        st.session_state.model_name = model_choice
        if st.session_state.engine:
            st.session_state.engine.llm.model_name = model_choice

    divider()

    # How to Use Popover
    st.markdown("### 💡 Help & Guides")
    with st.expander("📖 ZMET How-to-Use Guide"):
        st.write("""
        **Welcome to ZMET AI Moderator!**
        
        This application automates the Zaltman Metaphor Elicitation Technique (ZMET) research interview.
        
        **Workflow Steps:**
        1. **Storytelling:** Upload 3-6 pre-selected images that represent your thoughts and feelings about the topic. Provide a short story or narrative for each.
        2. **Missed Issues:** Describe any thoughts or feelings you couldn't find an image for.
        3. **Pile Sorting:** Group your images into piles. The AI will label themes.
        4. **Triads & Laddering:** The AI moderator conducts Kelly-Grid comparison. You will explain why concepts matter to explore Attributes, Consequences, and Values.
        5. **Sensory:** Select the overall most representative image and provide sensory analogies (taste, touch, smell, sound).
        6. **Consolidated Map & Vignette:** Review the resulting mental map and vignette. Download the researcher deliverables.
        """)

# Initialize ZMET Interview Engine if key is provided and engine does not exist
if st.session_state.api_key and not st.session_state.engine:
    llm_client = ZMETLLMClient(api_key=st.session_state.api_key, model_name=st.session_state.model_name)
    st.session_state.engine = ZMETInterviewEngine(llm_client)

# Guard against missing API Key
if not st.session_state.api_key:
    st.warning("⚠️ Please provide a Gemini API Key in the sidebar to start the interview session.")
    st.stop()

# Helper to check if step state changed and force refresh
engine = st.session_state.engine
session = engine.session

# =====================================================================
# 3. Main Dashboard Layout (Step Rendering)
# =====================================================================

st.title("🧠 ZMET Automated Interview Prototypes")
st.write("Conduct guided conversational user research powered by Gemini 2.5.")

# GATE: Research topic must be set before any step begins. Every AI prompt
# in interview_engine.py (storytelling, missed issues, sorting, laddering,
# mental map, vignette) reads session.research_topic and references it --
# without this gate the AI has no idea what it's interviewing about, which
# is why it defaults to generic, non-specific questions.
if not session.research_topic:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Before we begin: what are you researching?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="zmet-subheader">This grounds every question the AI moderator asks. '
            'Storytelling prompts, laddering probes, and the closing summary will all reference this topic '
            'instead of asking generically.</div>',
            unsafe_allow_html=True
        )

        topic_input = st.text_input(
            "Research topic",
            placeholder="e.g. 'intimate apparel', 'grocery shopping experience', 'commuting to work'",
            help="Be specific -- this is the subject the participant's images, stories, and laddering will all be about."
        )

        if st.button("Begin Interview ➔", disabled=not topic_input.strip()):
            session.research_topic = topic_input.strip()
            st.rerun()

    st.stop()

# Keep the active topic visible throughout the session so it's never
# ambiguous what the interview is grounded in.
st.caption(f"📌 Research topic: **{session.research_topic}**")

# Compact single-row step tracker (replaces the old persistent sidebar list)
_steps = [(3, "Storytelling"), (5, "Sorting"), (6, "Laddering"), (7, "Sensory"), (9, "Map"), (10, "Vignette")]
_pills = []
for _num, _label in _steps:
    if session.current_step > _num:
        _cls = "step-pill step-pill-done"
    elif session.current_step == _num:
        _cls = "step-pill step-pill-active"
    else:
        _cls = "step-pill"
    _pills.append(f'<span class="{_cls}">{_label}</span>')
st.markdown(f'<div class="step-tracker">{"".join(_pills)}</div>', unsafe_allow_html=True)

# STEP 3: Storytelling & Uploads
if session.current_step == 3:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 3 & 4: Storytelling & Missed Issues</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">Upload 3 to 6 images representing your feelings about the research topic. Then narrate the story of each image.</div>', unsafe_allow_html=True)
    
        # Image Uploader
        uploaded_files = st.file_uploader(
            "Upload Images (Min 3, Max 6)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Choose 3 to 6 images to compare. We use these for triad comparison later."
        )
    
        if uploaded_files:
            if len(uploaded_files) < 3:
                st.warning("⚠️ Kelly Grid Triads require at least 3 uploaded images. Please upload more.")
            elif len(uploaded_files) > 6:
                st.error("⚠️ Maximum 6 images allowed for this MVP.")
        
            # Display narration cards for uploaded files
            narrations = {}
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, file in enumerate(uploaded_files):
                col = cols[idx % 3]
                with col:
                    # Open image
                    image = Image.open(file)
                    st.image(image, use_container_width=True, caption=file.name)
                
                    # Narration Text Box
                    narrations[file.name] = st.text_area(
                        f"Narration for {file.name}",
                        key=f"narration_{file.name}",
                        placeholder="This image represents my feelings of...",
                        help="Describe the thoughts, stories, and feelings this picture evokes for you."
                    )
                
            # Missed Issues Text Input
            divider()
            missed_issues_text = st.text_area(
                "Step 4: Missed Issues & Images",
                placeholder="Are there any thoughts or feelings you couldn't find a picture for? Describe them in your own words...",
                help="ZMET Step 4: Describe any missing insights that you could not capture with visual images."
            )
        
            # Action button to trigger LLM Processing
            if st.button("Proceed to Pile Sorting ➔", disabled=(len(uploaded_files) < 3)):
                all_narrated = True
                for fname, val in narrations.items():
                    if not val.strip():
                        all_narrated = False
                        st.error(f"Please provide a narration for {fname}.")
            
                if all_narrated:
                    with st.spinner("Analyzing storytelling narrations and extracting ZMET attributes..."):
                        # Process images
                        for idx, file in enumerate(uploaded_files):
                            # Re-read image bytes
                            file.seek(0)
                            img_obj = Image.open(file)
                            engine.process_storytelling(
                                image_id=f"img_{idx+1:02d}",
                                filename=file.name,
                                narration=narrations[file.name],
                                image_obj=img_obj
                            )
                        
                        # Process missed issues
                        if missed_issues_text.strip():
                            engine.process_missed_issues(missed_issues_text)
                    
                        # Proceed step
                        session.current_step = 5
                        st.rerun()
                    

# STEP 5: Pile Sorting
elif session.current_step == 5:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 5: Thematic Pile Sorting</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">Group your uploaded images into piles that share similar themes. We will suggest thematic labels based on your stories.</div>', unsafe_allow_html=True)
    
        # We allow sorting into 2 default piles
        pile_options = ["Pile A", "Pile B", "Pile C"]
    
        # Store sorting maps
        sorting_map = {}
    
        cols = st.columns(len(session.images))
        for idx, img in enumerate(session.images):
            with cols[idx]:
                # Load filename (since bytes aren't stored, we display names or mock the image if needed)
                st.info(f"📷 {img.filename}")
                st.write(f"*{img.user_narration[:60]}...*")
            
                # Selectbox to select Pile
                selected_pile = st.selectbox(
                    f"Group Image {idx+1}",
                    options=pile_options,
                    key=f"pile_select_{img.image_id}",
                    help="Select the pile/theme category this image belongs to."
                )
                sorting_map[img.image_id] = selected_pile

        # Button to request suggested pile themes from Gemini
        divider()
        if st.button("Generate Pile Themes & Probes"):
            # Re-organize piles for engine call
            piles_data = {}
            for img_id, pile_name in sorting_map.items():
                piles_data.setdefault(pile_name, []).append(img_id)
            
            with st.spinner("AI analyzing groups to suggest themes..."):
                suggested_labels = engine.suggest_sorting_themes(piles_data)
                st.session_state[f"suggested_pile_labels"] = suggested_labels
            
        # Display suggested pile labels and allow override
        if f"suggested_pile_labels" in st.session_state:
            st.markdown("### Suggested Pile Themes")
            user_labels = {}
            for pile_name, default_label in st.session_state[f"suggested_pile_labels"].items():
                user_labels[pile_name] = st.text_input(
                    f"Theme Label for {pile_name}",
                    value=default_label,
                    key=f"theme_override_{pile_name}",
                    help="Edit this text to rename the theme category."
                )
            
            if st.button("Confirm Groups & Start Interview Chat ➔"):
                # Store groups in constructs list
                for pile_name, label in user_labels.items():
                    c_id = engine.get_next_construct_id()
                    session.constructs.append(Construct(
                        id=c_id,
                        name=f"Sorting Pile: {label}",
                        type="attribute",
                        source_step=5
                    ))
            
                session.current_step = 6
                st.rerun()
            

# STEP 6: Construct Elicitation & Laddering (The Core Dialogue Loop)
elif session.current_step == 6:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 6: Triad Elicitation & AI Laddering</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">The AI moderator compares visual elements to probe "Why does that matter to you?". This laddering chain uncovers deep values.</div>', unsafe_allow_html=True)
    
        # Initialize triad question if not active
        if not engine.active_ladder:
            with st.spinner("Moderator analyzing images and choosing a comparison triad..."):
                triad_q = engine.start_laddering_triad()
                st.session_state.triad_question = triad_q
            
        # Show active items in comparison
        if engine.active_ladder:
            st.markdown("#### Currently Comparing Items:")
            cols = st.columns(3)
            for idx, item in enumerate(engine.active_ladder.get("items", [])):
                with cols[idx]:
                    # If item is an image ID, display filename
                    img_meta = next((img for img in session.images if img.image_id == item), None)
                    if img_meta:
                        st.info(f"📷 {img_meta.filename}")
                        st.write(f"*{img_meta.user_narration[:80]}...*")
                    else:
                        st.warning(f"Concept: {item}")
                    
        # Chat display block
        divider()
    
        # Render chat history for Step 6
        st.markdown("💬 **Interview Dialogue:**")
        for msg in session.transcript[-6:]:  # Show last few messages for clean layout
            sender_label = "🤖 AI Moderator" if msg["sender"] == "moderator" else "👤 Respondent"
            bubble_cls = "chat-moderator" if msg["sender"] == "moderator" else "chat-respondent"
            st.markdown(
                f'<div class="{bubble_cls}"><strong>{sender_label}:</strong> {msg["text"]}</div>',
                unsafe_allow_html=True
            )
        
        # Text input for respondent reply
        respondent_reply = st.text_area(
            "Your Response",
            key="respondent_reply_input",
            placeholder="Type why this theme or attribute matters to you, or describe how they relate...",
            help="Respond to the AI moderator's question. Try to explain what thoughts or emotions are triggered."
        )
    
        if st.button("Send Reply", disabled=not respondent_reply.strip()):
            with st.spinner("Moderator evaluating response and advancing the ladder..."):
                next_q = engine.process_laddering_response(respondent_reply)
                st.session_state.triad_question = next_q
                st.rerun()
            

# STEP 7 & 8: Most Representative Image & Sensory Analogies
elif session.current_step == 7:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 7a & 8: Representative Focus & Sensory Analogies</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">Finalize the metaphor elicitation by picking a core focus image and describing sensory analogies.</div>', unsafe_allow_html=True)
    
        # Step 7a: Representative Image
        st.markdown("### Step 7a: Most Representative Image Selection")
        img_filenames = [img.filename for img in session.images]
        rep_filename = st.radio(
            "Select the single image that BEST summarizes your overall feelings on the topic:",
            options=img_filenames,
            help="ZMET Step 7a: Choose the one anchor image that contains the core metaphor for you."
        )
    
        rep_explanation = st.text_area(
            "Why does this specific image sum up your core feelings?",
            placeholder="Explain the overarching metaphor this image holds for you...",
            help="Provide a narrative details describing this core metaphor choice."
        )
    
        # Step 8: Sensory Analogies
        divider()
        st.markdown("### Step 8: Sensory Analogies")
        st.write("Without adding new images, describe sensory analogies that represent your feelings on the topic:")
    
        col1, col2 = st.columns(2)
        with col1:
            taste_val = st.text_input("Taste", placeholder="e.g. Clean spring water, bitter espresso...", help="What taste matches your feelings on the topic?")
            touch_val = st.text_input("Touch/Texture", placeholder="e.g. Smooth silk, rough bark...", help="What touch texture matches your feelings on the topic?")
        with col2:
            smell_val = st.text_input("Smell", placeholder="e.g. Fresh rain, pine woods...", help="What scent matches your feelings on the topic?")
            sound_val = st.text_input("Sound", placeholder="e.g. Distant thunder, quiet wind...", help="What sound/music matches your feelings on the topic?")
        
        if st.button("Complete Interview Elicitation ➔"):
            if not rep_explanation.strip():
                st.error("Please explain why you selected the representative image.")
            else:
                with st.spinner("Processing representative focus and sensory analogies..."):
                    # Save representative image details
                    rep_img = next((img for img in session.images if img.filename == rep_filename), None)
                    if rep_img:
                        engine.process_representative_image(rep_img.image_id, rep_explanation)
                    
                    # Save sensory
                    sensory_data = {
                        "taste": taste_val,
                        "touch": touch_val,
                        "smell": smell_val,
                        "sound": sound_val
                    }
                    engine.process_sensory_analogies(sensory_data)
                
                    # Advance step to Map consolidation
                    session.current_step = 9
                    st.rerun()
                

# STEP 9: Mental Map Generation
elif session.current_step == 9:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 9: Consolidated Mental Map</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">The AI consolidates duplicate constructs and renders an interactive, hierarchical network showing Attributes, Consequences, and Values.</div>', unsafe_allow_html=True)
    
        # Run map consolidation
        if "consolidated_map" not in st.session_state:
            with st.spinner("AI is consolidating constructs and constructing relationship network..."):
                cons_map = engine.generate_mental_map()
                st.session_state.consolidated_map = cons_map
            
        # Display the Pyvis network iframe
        html_graph = generate_pyvis_html(session.constructs, session.relations)
    
        st.markdown("### Interactive Network Visualizer")
        st.write("Drag nodes to organize. Hover over nodes to see classifications; hover over edges to read link rationales.")
        components.html(html_graph, height=520, scrolling=False)
    
        # Key mapping details
        st.markdown("""
        💡 **Legend:**
        *   🔵 **Boxes (Blue):** Attributes (concrete features or settings)
        *   🟡 **Ellipses (Amber):** Consequences (functional/psychosocial outcomes)
        *   🔴 **Large Dots (Pink/Rose):** Values (terminal emotional drivers)
        """)
    
        divider()
        if st.button("Proceed to Final Summary Vignette ➔"):
            session.current_step = 10
            st.rerun()
        

# STEP 10: Vignette & Final Exporter Dashboard (The "Done" State)
elif session.current_step == 10:
    with st.container(border=True):
        st.markdown('<div class="zmet-header">Step 10: Summary Vignette & Closing Narrative</div>', unsafe_allow_html=True)
        st.markdown('<div class="zmet-subheader">Final step: Compile a summary of prior insights and write a closing story.</div>', unsafe_allow_html=True)
    
        # Respondent input for final vignette story
        vignette_input = st.text_area(
            "Tell a brief closing story or vignette summarizing what this topic means to you, incorporating some of the attributes and values identified:",
            placeholder="In the end, this topic means that...",
            help="ZMET Step 10: Create a short story summarizing your overall metaphor network."
        )
    
        if st.button("Generate Final Deliverables"):
            if not vignette_input.strip():
                st.error("Please provide your closing story before generating deliverables.")
            else:
                with st.spinner("AI compiling final deliverables and analyzing consensus chains..."):
                    engine.generate_vignette(vignette_input)
                    st.session_state.completed = True
                    st.rerun()
                

# THE COMPLETED DONE STATE: RESEARCHER DASHBOARD
if st.session_state.get("completed", False):
    st.success("🎉 ZMET Interview Complete! Deliverables ready for the Researcher.")
    
    with st.container(border=True):
        st.markdown('<div class="zmet-header">📁 Researcher Analysis Dashboard</div>', unsafe_allow_html=True)
    
        tab1, tab2, tab3 = st.tabs(["📊 Hierarchical Map", "📝 Insights Vignette", "💬 Session Transcript"])
    
        with tab1:
            st.markdown("### Attribute-Consequence-Value Consensus Network")
            html_graph = generate_pyvis_html(session.constructs, session.relations)
            components.html(html_graph, height=520)
        
        with tab2:
            st.markdown("### Researcher Summary Report")
            st.markdown(f"**Respondent Summary Story (First-Person):**")
            st.info(session.vignette.summary_story)
        
            st.markdown(f"**Analytical Hierarchy Summary (Third-Person):**")
            st.write(session.vignette.researcher_summary)
        
            # Display constructs table
            st.markdown("#### Identified Constructs list")
            construct_rows = []
            for c in session.constructs:
                construct_rows.append({"ID": c.id, "Construct Name": c.name, "Hierarchy Classification": c.type.upper()})
            st.table(construct_rows)
        
        with tab3:
            st.markdown("### Sequential Dialog Log")
            for msg in session.transcript:
                sender_lbl = "🤖 Moderator" if msg["sender"] == "moderator" else "👤 Respondent"
                st.markdown(f"**{sender_lbl}:** {msg['text']}")
            
        # Exporter tools
        divider()
        st.markdown("### 📥 Exporter Actions")
    
        col_pdf, col_json = st.columns(2)
        with col_pdf:
            # Generate PDF Bytes
            with st.spinner("Compiling PDF report..."):
                try:
                    pdf_data = generate_zmet_pdf(session)
                    st.download_button(
                        "Download Complete Session PDF",
                        data=pdf_data,
                        file_name=f"zmet_report_{session.session_id}.pdf",
                        mime="application/pdf",
                        help="Downloads a structured, professional PDF report of the entire interview session."
                    )
                except Exception as e:
                    st.error(f"Error generating PDF report: {e}")
                
        with col_json:
            # Download JSON
            session_json = json.dumps(session.model_dump(), indent=2)
            st.download_button(
                "Download Complete Session JSON",
                data=session_json,
                file_name=f"zmet_session_{session.session_id}.json",
                mime="application/json",
                help="Downloads full session state including image metadata, constructs list, relationships, and transcripts."
            )
    
        # Reset Button for new session
        if st.button("Start New ZMET Session 🔄"):
            # Reset Session State
            st.session_state.engine = ZMETInterviewEngine(
                ZMETLLMClient(api_key=st.session_state.api_key, model_name=st.session_state.model_name)
            )
            if "consolidated_map" in st.session_state:
                del st.session_state.consolidated_map
            if "completed" in st.session_state:
                st.session_state.completed = False
            st.rerun()
        
