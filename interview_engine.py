import os
import io
import base64
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from PIL import Image

# Import LLM client helper
from llm_client import ZMETLLMClient

logger = logging.getLogger("ZMETInterviewEngine")

# A real means-end chain should pass through at least one consequence
# before reaching a value (attribute -> consequence -> value = depth 3).
# MIN_LADDER_DEPTH is the floor below which a terminal value is never
# honored, even if the model tries to declare one early. MAX_LADDER_DEPTH
# is a hard safety cap so a chain can't loop indefinitely if the model
# keeps refusing to terminate.
MIN_LADDER_DEPTH = 3
MAX_LADDER_DEPTH = 5

# =====================================================================
# 1. ZMET Session Data Schema (Matches proposed JSON schema)
# =====================================================================

class Construct(BaseModel):
    id: str
    name: str
    type: str  # "attribute", "consequence", "value"
    source_step: int
    source_image_id: Optional[str] = None

class Relation(BaseModel):
    relation_id: str
    source: str  # Construct ID
    target: str  # Construct ID
    link_type: str = "leads_to"
    explanation: str

class SensoryAnalogies(BaseModel):
    taste: Optional[str] = None
    touch: Optional[str] = None
    smell: Optional[str] = None
    sound: Optional[str] = None

class Vignette(BaseModel):
    summary_story: Optional[str] = None
    researcher_summary: Optional[str] = None

class ImageMetadata(BaseModel):
    image_id: str
    filename: str
    user_narration: str
    representative: bool = False
    # Base64-encoded image bytes. Stored (not just referenced by filename) so
    # the actual image can be re-attached to later Gemini calls, e.g. the
    # Step 6 triad/laddering comparison, which needs the real visual content
    # rather than just the filename or narration text.
    image_b64: Optional[str] = None

class ZMETSession(BaseModel):
    session_id: str
    started_at: str
    completed_at: Optional[str] = None
    respondent_id: str
    current_step: int = 3
    # The subject of the interview (e.g. "intimate apparel", "grocery
    # shopping experience"). Set once at the start and referenced in every
    # AI prompt below so questions are specific to this topic rather than
    # generic ZMET boilerplate.
    research_topic: str = ""
    images: List[ImageMetadata] = []
    # IDs of images that have already been used in a completed Step 6 triad
    # comparison. select_triad() reads this so subsequent calls pick a
    # fresh, not-yet-compared set of images instead of always returning the
    # first three uploaded.
    laddered_image_ids: List[str] = []
    constructs: List[Construct] = []
    relations: List[Relation] = []
    sensory_analogies: SensoryAnalogies = Field(default_factory=SensoryAnalogies)
    vignette: Vignette = Field(default_factory=Vignette)
    transcript: List[Dict[str, str]] = []

# =====================================================================
# 2. Pydantic Models for Structured Gemini Outputs
# =====================================================================

class ExtractedConstruct(BaseModel):
    name: str = Field(description="Name of the concept/construct (e.g. 'Quiet Environment', 'Peace of Mind')")
    type: str = Field(description="Type of the construct: must be one of 'attribute', 'consequence', or 'value'")
    explanation: str = Field(description="Brief explanation of why this construct was identified from the text")

class ExtractedRelation(BaseModel):
    source_name: str = Field(description="Name of the source construct, exactly matching one of the 'name' values in the constructs list above")
    target_name: str = Field(description="Name of the target construct that the source leads to, exactly matching one of the 'name' values in the constructs list above")
    explanation: str = Field(description="Brief explanation of the causal link, e.g. 'Open space layout leads to feeling of freedom'")

class ExtractedConstructsList(BaseModel):
    constructs: List[ExtractedConstruct]
    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        description="Any causal links between the constructs above that the participant's own words imply (e.g. an attribute leading to a consequence, mentioned in the same narration). Leave empty only if a single construct was found or no link is evident -- do not force a relation that isn't actually implied."
    )

class SuggestedTheme(BaseModel):
    pile_id: str
    suggested_label: str
    explanation: str

class SuggestedThemesList(BaseModel):
    themes: List[SuggestedTheme]

class TriadExtractionResult(BaseModel):
    similarity_construct: str = Field(description="The construct (Attribute) shared by the two similar images, e.g. 'Natural landscape' or 'High energy'.")
    difference_construct: str = Field(description="The construct (Attribute) that sets the third image apart, e.g. 'Industrial setting' or 'Passive state'.")
    similar_image_ids: List[str] = Field(description="The IDs of the two similar images.")
    different_image_id: str = Field(description="The ID of the different image.")
    next_question: str = Field(description="The first laddering question probing the similarity construct, e.g., 'You mentioned two images show natural landscapes. Why is having a natural landscape important or meaningful to you?'")

class LadderStepResult(BaseModel):
    new_construct_name: Optional[str] = Field(None, description="The name of the new construct extracted, or null if no new construct is found.")
    new_construct_type: Optional[str] = Field(None, description="Must be one of 'attribute', 'consequence', or 'value', or null.")
    relation_explanation: Optional[str] = Field(None, description="Brief explanation of how the previous construct leads to this new construct.")
    is_terminal_value: bool = Field(description="True if the response indicates we have reached a core terminal value (e.g., self-actualization, deep peace, security) and further 'why' questions are unnecessary or circular.")
    next_question: str = Field(description="The next follow-up question the moderator should ask. If is_terminal_value is True, this should be a transition question or wrap-up of the concept chain.")

class UnifiedConstruct(BaseModel):
    id: str = Field(description="A clean ID like 'c_01'")
    name: str = Field(description="Consolidated, clean name of the construct")
    type: str = Field(description="Must be 'attribute', 'consequence', or 'value'")

class UnifiedRelation(BaseModel):
    source_id: str = Field(description="The source construct ID")
    target_id: str = Field(description="The target construct ID")
    explanation: str = Field(description="A brief sentence explaining the link (e.g. 'Nature leads to relaxation')")

class ConsolidatedMentalMap(BaseModel):
    constructs: List[UnifiedConstruct]
    relations: List[UnifiedRelation]

# =====================================================================
# 3. ZMET Interview Engine State Machine
# =====================================================================

class ZMETInterviewEngine:
    """Manages the interview state machine, prompting templates, and construct database."""
    
    def __init__(self, llm_client: ZMETLLMClient, session: Optional[ZMETSession] = None):
        self.llm = llm_client
        self.session = session or ZMETSession(
            session_id=f"zmet_session_{uuid.uuid4().hex[:8]}",
            started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            respondent_id="respondent_user"
        )
        
        # Internal state for tracking current laddering process in Step 6
        # Format: { 'triad_images': [...], 'current_construct_id': 'c_x', 'depth': 0, 'ladder_chain': [...] }
        self.active_ladder: Dict[str, Any] = {}

    def add_to_transcript(self, sender: str, text: str):
        """Append a message to the session transcript."""
        self.session.transcript.append({
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sender": sender,
            "text": text
        })

    def get_construct_by_id(self, construct_id: str) -> Optional[Construct]:
        """Find a construct by its ID."""
        for c in self.session.constructs:
            if c.id == construct_id:
                return c
        return None

    def get_next_construct_id(self) -> str:
        """Helper to generate sequential construct IDs."""
        return f"c_{len(self.session.constructs) + 1:02d}"

    def get_next_relation_id(self) -> str:
        """Helper to generate sequential relation IDs."""
        return f"r_{len(self.session.relations) + 1:02d}"

    @staticmethod
    def _encode_image(image_obj: Image.Image) -> Optional[str]:
        """Encode a PIL image to a base64 string for storage in ImageMetadata."""
        try:
            buf = io.BytesIO()
            fmt = (image_obj.format or "PNG").upper()
            # JPEG can't save an image with an alpha channel; fall back to PNG.
            if fmt == "JPEG" and image_obj.mode in ("RGBA", "P"):
                fmt = "PNG"
            image_obj.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image for storage: {e}")
            return None

    def _load_image(self, image_id: str) -> Optional[Image.Image]:
        """Reload a previously stored image (by ID) as a PIL Image, so it can
        be re-attached to a Gemini call in a later step (e.g. Step 6)."""
        img_meta = next((img for img in self.session.images if img.image_id == image_id), None)
        if not img_meta or not img_meta.image_b64:
            return None
        try:
            raw = base64.b64decode(img_meta.image_b64)
            return Image.open(io.BytesIO(raw))
        except Exception as e:
            logger.error(f"Failed to decode stored image '{image_id}': {e}")
            return None

    # -----------------------------------------------------------------
    # Step 3: Storytelling (Story Extraction & Construct Elicitation)
    # -----------------------------------------------------------------
    def process_storytelling(self, image_id: str, filename: str, narration: str, image_obj: Optional[Image.Image] = None) -> List[Construct]:
        """
        Receives an uploaded image narration, uses Gemini Vision to understand the image + text,
        extracts initial constructs (primarily attributes), and appends them to session.
        """
        # Encode and persist the image bytes so later steps (e.g. Step 6
        # laddering) can re-attach the real image, not just its filename.
        encoded_image = self._encode_image(image_obj) if image_obj is not None else None

        # Add to image list if not already present
        existing_img = next((img for img in self.session.images if img.image_id == image_id), None)
        if not existing_img:
            img_meta = ImageMetadata(
                image_id=image_id,
                filename=filename,
                user_narration=narration,
                image_b64=encoded_image
            )
            self.session.images.append(img_meta)
        else:
            existing_img.user_narration = narration
            if encoded_image:
                existing_img.image_b64 = encoded_image

        # Generate prompt for construct extraction
        topic_line = f'This interview is specifically about: "{self.session.research_topic}".' if self.session.research_topic else ""
        prompt = f"""
        You are a professional ZMET (Zaltman Metaphor Elicitation Technique) research analyst.
        {topic_line}
        The participant uploaded an image representing their thoughts/feelings about this topic and narrated the following story:
        "{narration}"

        Analyze the narration and the visual context of the image, specifically in relation to the research topic above. Extract 1 to 3 primary constructs.
        IMPORTANT: Do NOT copy the user's verbatim phrasing. Translate and abstract their comments into standardized, clean academic construct names (e.g., translate 'I don't have to look over my shoulder' to the Consequence 'Security/Safety' or 'Anxiety Reduction'; translate 'It feels so open' to the Attribute 'Open Space Layout').
        
        Classify each construct strictly into one of the ZMET categories:
        - attribute: Concrete or abstract visual/functional properties of the subject.
        - consequence: Functional or psychosocial outcomes resulting from those attributes.
        - value: Core terminal emotional drivers or life goals (e.g., Security, Freedom, Peace of Mind, Achievement, Connection, Self-Esteem).

        If the narration implies that one construct leads to another (e.g. an attribute causing a consequence), also report that as a relation between their exact names. Only report a relation if it's genuinely implied by what the participant said -- don't force one if only a single construct was found.

        Return the results in the requested JSON structure.
        """
        
        try:
            # We call Gemini with the visual context of the image + narration text
            raw_response = self.llm.generate_content(
                prompt=prompt,
                image=image_obj,
                response_schema=ExtractedConstructsList
            )
            
            parsed = ExtractedConstructsList.model_validate_json(raw_response)
            new_constructs = []
            name_to_id: Dict[str, str] = {}
            for item in parsed.constructs:
                c_id = self.get_next_construct_id()
                new_c = Construct(
                    id=c_id,
                    name=item.name,
                    type=item.type.lower(),
                    source_step=3,
                    source_image_id=image_id
                )
                self.session.constructs.append(new_c)
                new_constructs.append(new_c)
                name_to_id[item.name] = c_id

            # Persist any causal links the model found between the constructs
            # it just extracted, so this image's constructs aren't left as
            # isolated nodes in the eventual mental map.
            for rel in parsed.relations:
                src_id = name_to_id.get(rel.source_name)
                tgt_id = name_to_id.get(rel.target_name)
                if src_id and tgt_id and src_id != tgt_id:
                    self.session.relations.append(Relation(
                        relation_id=self.get_next_relation_id(),
                        source=src_id,
                        target=tgt_id,
                        explanation=rel.explanation
                    ))

            return new_constructs
        except Exception as e:
            # Fallback in case of parsing/API errors -- logged so a dead model
            # or bad schema shows up as a real error instead of silently
            # degrading to generic placeholder constructs.
            logger.error(f"process_storytelling failed for image '{image_id}': {e}", exc_info=True)
            c_id = self.get_next_construct_id()
            fallback_c = Construct(
                id=c_id,
                name="Visual Story Theme",
                type="attribute",
                source_step=3,
                source_image_id=image_id
            )
            self.session.constructs.append(fallback_c)
            return [fallback_c]

    # -----------------------------------------------------------------
    # Step 4: Missed Issues & Images
    # -----------------------------------------------------------------
    def process_missed_issues(self, text: str) -> List[Construct]:
        """
        Process the text for thoughts or feelings that the participant couldn't find a picture for,
        extracting constructs and adding them to the state.
        """
        if not text.strip():
            return []

        topic_line = f'This interview is specifically about: "{self.session.research_topic}".' if self.session.research_topic else ""
        prompt = f"""
        You are a ZMET research analyst. {topic_line}
        The participant described thoughts or feelings related to this topic that they couldn't find a picture for:
        "{text}"

        Analyze this text and extract 1 to 3 primary ZMET constructs. Do not copy their words verbatim. Abstract and translate their comments into standardized research construct names.
        Classify each construct type as 'attribute', 'consequence', or 'value' (e.g., Security, Peace of Mind, Achievement).
        If the text implies one construct leads to another, also report that as a relation between their exact names. Only report a relation if it's genuinely implied.
        Return them in the requested JSON structure.
        """
        try:
            raw_response = self.llm.generate_content(
                prompt=prompt,
                response_schema=ExtractedConstructsList
            )
            parsed = ExtractedConstructsList.model_validate_json(raw_response)
            new_constructs = []
            name_to_id: Dict[str, str] = {}
            for item in parsed.constructs:
                c_id = self.get_next_construct_id()
                new_c = Construct(
                    id=c_id,
                    name=item.name,
                    type=item.type.lower(),
                    source_step=4
                )
                self.session.constructs.append(new_c)
                new_constructs.append(new_c)
                name_to_id[item.name] = c_id

            for rel in parsed.relations:
                src_id = name_to_id.get(rel.source_name)
                tgt_id = name_to_id.get(rel.target_name)
                if src_id and tgt_id and src_id != tgt_id:
                    self.session.relations.append(Relation(
                        relation_id=self.get_next_relation_id(),
                        source=src_id,
                        target=tgt_id,
                        explanation=rel.explanation
                    ))

            return new_constructs
        except Exception as e:
            logger.error(f"process_missed_issues failed: {e}", exc_info=True)
            c_id = self.get_next_construct_id()
            fallback_c = Construct(
                id=c_id,
                name="Missed Theme",
                type="attribute",
                source_step=4
            )
            self.session.constructs.append(fallback_c)
            return [fallback_c]

    # -----------------------------------------------------------------
    # Step 5: Sorting Task
    # -----------------------------------------------------------------
    def suggest_sorting_themes(self, piles: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Based on user grouped image IDs, query Gemini to suggest a label/theme for each pile
        based on the user's previously provided storytelling narrations for those images.
        """
        # Build prompt using image narrations
        pile_descriptions = []
        for pile_name, img_ids in piles.items():
            narrations = []
            for img_id in img_ids:
                img_meta = next((img for img in self.session.images if img.image_id == img_id), None)
                if img_meta:
                    narrations.append(f"- Image {img_meta.filename}: '{img_meta.user_narration}'")
            pile_descriptions.append(
                f"Pile ID: {pile_name}\n" + "\n".join(narrations)
            )

        topic_line = f'This interview is specifically about: "{self.session.research_topic}".\n' if self.session.research_topic else ""
        prompt = f"""
        {topic_line}The participant has sorted their uploaded images into piles. Below are the images in each pile and what the participant said about them:

        {"\n\n".join(pile_descriptions)}

        Based on these descriptions and the research topic above, suggest a single concise thematic label for each Pile.
        Provide the response in the requested JSON structure.
        """
        
        try:
            raw_response = self.llm.generate_content(
                prompt=prompt,
                response_schema=SuggestedThemesList
            )
            parsed = SuggestedThemesList.model_validate_json(raw_response)
            return {theme.pile_id: theme.suggested_label for theme in parsed.themes}
        except Exception as e:
            logger.error(f"suggest_sorting_themes failed: {e}", exc_info=True)
            # Fallback default themes
            return {pile_name: f"Group Theme {idx+1}" for idx, pile_name in enumerate(piles.keys())}

    # -----------------------------------------------------------------
    # Step 6: Construct Elicitation & Laddering (The Core Elicitation Loop)
    # -----------------------------------------------------------------
    def select_triad(self) -> Optional[List[ImageMetadata]]:
        """Select the next triad (3) of images that haven't been compared yet.
        Cycles through all uploaded images in groups of 3, rather than
        always returning the same first three -- otherwise images 4+ never
        go through Kelly Grid comparison or laddering at all, leaving their
        constructs permanently disconnected from the reasoning chain."""
        remaining = [img for img in self.session.images if img.image_id not in self.session.laddered_image_ids]
        if len(remaining) >= 3:
            return remaining[:3]
        return None

    def start_laddering_triad(self) -> str:
        """
        Initiates Step 6. Selects three images, generates the triad question,
        and initializes active laddering state.
        """
        triad = self.select_triad()
        if not triad:
            # Fallback if less than 3 images: combine images + text constructs
            concepts = [c.name for c in self.session.constructs[:3]]
            if len(concepts) < 3:
                return "We need at least 3 uploaded images or thoughts to perform comparison. Please describe some more details first."
            
            self.active_ladder = {
                "items": concepts,
                "is_text_triad": True,
                "depth": 0,
                "ladder_chain": [],
                "current_construct_id": None
            }
            topic_phrase = f"about {self.session.research_topic}" if self.session.research_topic else "here"
            question = f"Let's compare three of the concepts you mentioned {topic_phrase}: '{concepts[0]}', '{concepts[1]}', and '{concepts[2]}'. In what way are two of these similar, yet different from the third?"
            self.add_to_transcript("moderator", question)
            return question

        # Visual image triad
        # Mark these images as used immediately (not after the ladder chain
        # finishes) so a failure mid-chain doesn't cause the same triad to
        # be re-selected forever.
        self.session.laddered_image_ids.extend([img.image_id for img in triad])

        image_names = [img.filename for img in triad]
        self.active_ladder = {
            "items": [img.image_id for img in triad],
            "is_text_triad": False,
            "depth": 0,
            "ladder_chain": [],
            "current_construct_id": None
        }
        
        topic_phrase = f"about {self.session.research_topic}" if self.session.research_topic else ""
        question = f"Look at these three images: '{image_names[0]}', '{image_names[1]}', and '{image_names[2]}'. In what way are two of these similar, yet different from the third, regarding your feelings {topic_phrase}?"
        self.add_to_transcript("moderator", question)
        return question

    def process_laddering_response(self, user_text: str) -> str:
        """
        Main recursive handler for laddering.
        1. If first step (depth=0), extract the initial Attribute construct from triad comparison.
        2. If subsequent step, run a 'why is this important' probe to extract the next Consequence/Value construct.
        3. Save relationships in local state, and return the next question.
        """
        self.add_to_transcript("respondent", user_text)
        
        # Depth 0: First response to triad
        if self.active_ladder.get("depth", 0) == 0:
            # Reload the actual triad images (if this is a visual triad, not a
            # text-concept triad) so Gemini can genuinely compare the images
            # rather than reasoning from filenames/IDs alone.
            triad_images: List[Image.Image] = []
            if not self.active_ladder.get("is_text_triad", True):
                for item_id in self.active_ladder.get("items", []):
                    img = self._load_image(item_id)
                    if img:
                        triad_images.append(img)
                    else:
                        logger.warning(f"No stored image bytes found for '{item_id}' during laddering triad.")

            item_labels = self.active_ladder['items']
            visual_note = (
                "The three images are attached to this message in the same order as listed above."
                if triad_images else
                "No image bytes were available, so base this only on the participant's own description."
            )

            topic_line = f'This interview is specifically about: "{self.session.research_topic}".' if self.session.research_topic else ""
            prompt = f"""
            You are a professional ZMET researcher conducting Kelly Repertory Grid elicitation.
            {topic_line}
            The participant was asked how two of these items: {item_labels} are similar, yet different from the third, in relation to this research topic.
            {visual_note}
            Their response was:
            "{user_text}"

            Analyze their response (and the attached images, if present) in the context of the research topic above, and extract:
            1. The similarity construct (an Attribute: abstract and translate their description into a standardized construct name, e.g., 'Aesthetic Comfort' or 'Functional Efficiency').
            2. The difference construct (an Attribute: e.g., 'Chaotic Environment' or 'Isolation').
            3. The first laddering question to probe this similarity construct to uncover its consequence -- phrase it specifically, referencing the research topic where natural (e.g., 'You mentioned these feel [similarity construct] when it comes to {self.session.research_topic or "this"}. Why is that important or meaningful to you?').
            
            Return this strictly in the requested JSON structure.
            """
            
            try:
                raw_response = self.llm.generate_content(
                    prompt=prompt,
                    image=triad_images if triad_images else None,
                    response_schema=TriadExtractionResult
                )
                result = TriadExtractionResult.model_validate_json(raw_response)
                
                # Store the similarity construct in session
                c_id = self.get_next_construct_id()
                new_c = Construct(
                    id=c_id,
                    name=result.similarity_construct,
                    type="attribute",
                    source_step=6
                )
                self.session.constructs.append(new_c)
                
                # Update ladder state
                self.active_ladder["depth"] = 1
                self.active_ladder["current_construct_id"] = c_id
                self.active_ladder["ladder_chain"] = [c_id]
                
                self.add_to_transcript("moderator", result.next_question)
                return result.next_question
                
            except Exception as e:
                logger.error(f"Depth-0 laddering (triad extraction) failed: {e}", exc_info=True)
                # Fallback in case of failure
                c_id = self.get_next_construct_id()
                fallback_c = Construct(id=c_id, name="Shared Construct", type="attribute", source_step=6)
                self.session.constructs.append(fallback_c)
                self.active_ladder["depth"] = 1
                self.active_ladder["current_construct_id"] = c_id
                self.active_ladder["ladder_chain"] = [c_id]
                
                fallback_q = f"Why is '{fallback_c.name}' important or meaningful to you?"
                self.add_to_transcript("moderator", fallback_q)
                return fallback_q

        # Depth >= 1: Probing "Why does that matter to you?"
        prev_construct_id = self.active_ladder["current_construct_id"]
        prev_construct = self.get_construct_by_id(prev_construct_id)
        prev_name = prev_construct.name if prev_construct else "the previous topic"

        current_depth = self.active_ladder["depth"]
        # A proper means-end chain should pass through at least one
        # consequence before reaching a value (attribute -> consequence ->
        # value = depth 3). Without this floor, the model tends to declare
        # the very first "why" answer a terminal value and stop -- which is
        # exactly what produced only one laddering question per triad.
        depth_instruction = (
            f"This chain is currently at depth {current_depth} of a required minimum depth of {MIN_LADDER_DEPTH}. "
            f"Do NOT set is_terminal_value to True yet, even if the response sounds value-like -- instead, dig one "
            f"level deeper by asking why THIS specific new construct matters, to find the more fundamental driver "
            f"underneath it."
            if current_depth < MIN_LADDER_DEPTH else
            f"This chain has reached depth {current_depth}, meeting the minimum required depth. You may now set "
            f"is_terminal_value to True if the response genuinely represents a core terminal value with no "
            f"meaningful deeper 'why' -- but keep laddering if there's clearly one more real step underneath it."
        )

        topic_line = f'This interview is specifically about: "{self.session.research_topic}".' if self.session.research_topic else ""
        prompt = f"""
        You are a ZMET research interviewer executing means-end chain laddering. {topic_line}
        We are laddering from the construct: '{prev_name}' (type: {prev_construct.type if prev_construct else 'attribute/consequence'}), in the context of the research topic above.
        The participant was asked why that is important/meaningful to them. Their response:
        "{user_text}"

        {depth_instruction}

        Evaluate this response:
        1. Identify any new construct mentioned or heavily implied (type: 'consequence' or 'value').
           IMPORTANT: Do not copy the user's verbatim phrase. Translate and map their statement to a clean, standardized academic construct.
           (e.g., translate 'I feel trapped' to 'Consequence: Physical Imprisonment'; translate 'I don't have to worry about my boss' to 'Consequence: Anxiety Reduction').
        2. Categorize it as 'consequence' or 'value'. If it is a core emotional value or life goal (such as Security, Freedom/Independence, Peace of Mind, Achievement/Success, Belonging/Connection, Self-Esteem/Confidence), classify it as 'value'.
        3. Set is_terminal_value according to the depth rule above.
        4. Formulate the next probing question, grounded in the research topic. If is_terminal_value is False, ask something like 'Why is [new construct] important or meaningful to you when it comes to {self.session.research_topic or "this"}?' to climb higher up the ladder -- make it specific to the new construct just identified, not a repeat of the previous question. If is_terminal_value is True, formulate a transition question wrapping up this ladder chain.

        Return this strictly in the requested JSON structure.
        """
        
        try:
            raw_response = self.llm.generate_content(
                prompt=prompt,
                response_schema=LadderStepResult
            )
            result = LadderStepResult.model_validate_json(raw_response)
            
            if result.new_construct_name:
                c_id = self.get_next_construct_id()
                new_c = Construct(
                    id=c_id,
                    name=result.new_construct_name,
                    type=(result.new_construct_type or "consequence").lower(),
                    source_step=6
                )
                self.session.constructs.append(new_c)
                
                # Add relationship connection
                r_id = self.get_next_relation_id()
                rel = Relation(
                    relation_id=r_id,
                    source=prev_construct_id,
                    target=c_id,
                    link_type="leads_to",
                    explanation=result.relation_explanation or f"Participant notes that {prev_name} leads to {result.new_construct_name}."
                )
                self.session.relations.append(rel)
                
                # Update ladder state
                self.active_ladder["current_construct_id"] = c_id
                self.active_ladder["ladder_chain"].append(c_id)
            
            # Increment depth
            self.active_ladder["depth"] += 1
            
            # Check terminal condition. is_terminal_value from the model is
            # only honored once the minimum depth has actually been reached
            # -- this is the hard backstop for the "only one laddering
            # question" problem: even if the model tries to call something
            # terminal after a single hop, we don't let the chain end until
            # it's passed through enough levels for a real attribute ->
            # consequence -> value chain.
            reached_min_depth = self.active_ladder["depth"] >= MIN_LADDER_DEPTH
            if (result.is_terminal_value and reached_min_depth) or self.active_ladder["depth"] >= MAX_LADDER_DEPTH:
                # This ladder chain is complete. If more images haven't been
                # through a triad yet, start the next one instead of ending
                # the whole laddering phase after just one thread -- this is
                # what previously left images 4+ (and their constructs)
                # completely disconnected from the mental map.
                self.active_ladder = {}
                if self.select_triad() is not None:
                    next_question = self.start_laddering_triad()
                    return next_question

                self.session.current_step = 7  # Step 7a & 8 next
                transition_q = "Thank you. This concludes our triad comparisons. Let's look closer at your images now. Which of your uploaded images best represents your core feeling overall, and why?"
                self.add_to_transcript("moderator", transition_q)
                return transition_q
                
            self.add_to_transcript("moderator", result.next_question)
            return result.next_question
            
        except Exception as e:
            logger.error(f"Depth>=1 laddering probe failed: {e}", exc_info=True)
            # Fallback closure if API fails
            self.session.current_step = 7
            self.active_ladder = {}
            fallback_q = "Thank you. Let's move to the next phase. Which of your uploaded images best represents your core feeling overall, and why?"
            self.add_to_transcript("moderator", fallback_q)
            return fallback_q

    # -----------------------------------------------------------------
    # Step 7a & 8: Representative Image & Sensory Analogies
    # -----------------------------------------------------------------
    def process_representative_image(self, image_id: str, explanation: str):
        """Save the chosen most representative image and narration context."""
        for img in self.session.images:
            if img.image_id == image_id:
                img.representative = True
                
        # Add to transcript
        self.add_to_transcript("respondent", f"Representative Image: {image_id}. Explanation: {explanation}")
        
        # Add a construct representation
        c_id = self.get_next_construct_id()
        self.session.constructs.append(Construct(
            id=c_id,
            name=f"Core Metaphor Focus ({explanation[:30]}...)",
            type="consequence",
            source_step=7,
            source_image_id=image_id
        ))

    def process_sensory_analogies(self, sensory_data: Dict[str, str]):
        """Save sensory analogies (taste, touch, smell, sound) to state."""
        self.session.sensory_analogies = SensoryAnalogies(**sensory_data)
        
        # Log to transcript
        analogies_str = ", ".join([f"{k}: '{v}'" for k, v in sensory_data.items() if v])
        self.add_to_transcript("respondent", f"Sensory Analogies - {analogies_str}")
        
        # Add sensory constructs
        for sensory_type, description in sensory_data.items():
            if description:
                c_id = self.get_next_construct_id()
                self.session.constructs.append(Construct(
                    id=c_id,
                    name=f"Sensory Analogy ({sensory_type}): {description}",
                    type="attribute",
                    source_step=8
                ))

    # -----------------------------------------------------------------
    # Step 9: Mental Map (Construct Consolidation)
    # -----------------------------------------------------------------
    def generate_mental_map(self) -> Dict[str, Any]:
        """
        Calls Gemini to clean up, consolidate duplicates, and output a clean
        hierarchical network structure {constructs, relations} using JSON mode.
        """
        # Compile existing list of constructs and relationships
        constructs_desc = "\n".join([f"- ID: {c.id}, Name: '{c.name}', Type: '{c.type}'" for c in self.session.constructs])
        relations_desc = "\n".join([f"- {r.source} -> {r.target} (Reason: {r.explanation})" for r in self.session.relations])

        topic_line = f'This ZMET interview was conducted on the research topic: "{self.session.research_topic}".' if self.session.research_topic else ""
        prompt = f"""
        You are a research analyst summarizing a ZMET interview. {topic_line}
        Below are all the constructs and links generated dynamically during the respondent's interview:

        CONSTRUCTS:
        {constructs_desc}

        RELATIONSHIPS (EDGES):
        {relations_desc}

        Tasks:
        1. Consolidate constructs that mean the same thing (e.g., 'Quiet space' and 'Quiet environment') into a single construct.
        2. Assign them new sequential IDs starting with 'c_01', 'c_02', etc.
        3. Make sure their type ('attribute', 'consequence', or 'value') is clean and correct.
        4. Re-map the existing relationships (edges) above to use the new unified construct IDs. Keep link explanations clear and brief.
        5. IMPORTANT: Some constructs above may have no relationship connecting them to anything else -- they were captured from different parts of the interview (storytelling, sorting, sensory analogies, etc.) that didn't explicitly link back to the main reasoning chain. For every construct that would otherwise end up disconnected, infer a plausible causal link to the most semantically related construct already in the network (attribute -> consequence -> value direction), based on their meaning and the research topic. Add these as new relations alongside the re-mapped ones from step 4. Only skip this if a construct genuinely has no plausible connection to anything else -- the goal is a single connected means-end network, not scattered fragments, wherever the meaning genuinely supports it.
        
        Return the final consolidated constructs and relationships strictly in the requested JSON structure.
        """
        
        try:
            raw_response = self.llm.generate_content(
                prompt=prompt,
                response_schema=ConsolidatedMentalMap
            )
            parsed = ConsolidatedMentalMap.model_validate_json(raw_response)
            
            # Update our session store with the unified map
            self.session.constructs = [
                Construct(id=c.id, name=c.name, type=c.type, source_step=9)
                for c in parsed.constructs
            ]
            self.session.relations = [
                Relation(relation_id=f"r_{idx+1:02d}", source=r.source_id, target=r.target_id, explanation=r.explanation)
                for idx, r in enumerate(parsed.relations)
            ]
            
            return parsed.model_dump()
        except Exception as e:
            logger.error(f"generate_mental_map consolidation failed: {e}", exc_info=True)
            # If consolidating fails, return current map as fallback
            return {
                "constructs": [c.model_dump() for c in self.session.constructs],
                "relations": [r.model_dump() for r in self.session.relations]
            }

    # -----------------------------------------------------------------
    # Step 10: Summary Image & Vignette
    # -----------------------------------------------------------------
    def generate_vignette(self, closing_story: str) -> Vignette:
        """
        Compiles all narrative storytelling, sensory analogies, and the consolidated map
        into a written summary vignette of the respondent's core insights.
        """
        self.add_to_transcript("respondent", f"Closing Vignette Story: {closing_story}")
        
        # Build context
        narrations = "\n".join([f"- Image {img.filename}: {img.user_narration}" for img in self.session.images])
        map_pathway = "\n".join([f"- {c.name} ({c.type})" for c in self.session.constructs])
        
        topic_line = f'This interview explored the research topic: "{self.session.research_topic}".' if self.session.research_topic else ""
        prompt = f"""
        You are a ZMET research analyst. Compile the closing summary report. {topic_line}
        Here is the respondent's data:
        
        IMAGE NARRATIONS:
        {narrations}
        
        CONSTRUCTS DETECTED:
        {map_pathway}
        
        RESPONDENT'S CLOSING STORY:
        "{closing_story}"
        
        Compile:
        1. 'summary_story': A clean, consolidated first-person narrative story synthesizing their feelings.
        2. 'researcher_summary': An analytical third-person summary showing the Attributes -> Consequences -> Values chain.
        
        Return this in the requested JSON structure.
        """
        
        try:
            raw_response = self.llm.generate_content(
                prompt=prompt,
                response_schema=Vignette
            )
            parsed = Vignette.model_validate_json(raw_response)
            
            self.session.vignette = parsed
            self.session.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return parsed
        except Exception as e:
            logger.error(f"generate_vignette failed: {e}", exc_info=True)
            fallback = Vignette(
                summary_story=closing_story,
                researcher_summary=f"Respondent completed ZMET interview exploring {len(self.session.constructs)} constructs."
            )
            self.session.vignette = fallback
            self.session.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return fallback
