import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Import local data model types for hints (optional, but clean)
from interview_engine import ZMETSession

def generate_zmet_pdf(session: ZMETSession) -> bytes:
    """
    Generates a structured, clean, professional PDF report of a completed ZMET interview.
    Returns the PDF content as a bytes object.
    """
    buffer = BytesIO()
    
    # Page setup - 0.75 in margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom SaaS-like Styles
    primary_color = colors.HexColor("#1e293b")  # Dark Slate
    secondary_color = colors.HexColor("#3b82f6")  # Accent Blue
    text_color = colors.HexColor("#334155")
    bg_light = colors.HexColor("#f8fafc")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    vignette_style = ParagraphStyle(
        'Vignette_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10.5,
        leading=15,
        textColor=primary_color,
        spaceAfter=12
    )

    story = []
    
    # --- PAGE 1: HEADER & DELIVERABLES SUMMARY ---
    story.append(Paragraph("ZMET Metaphor Research Interview Report", title_style))
    
    meta_info = f"""
    <b>Session ID:</b> {session.session_id}<br/>
    <b>Respondent ID:</b> {session.respondent_id}<br/>
    <b>Started At:</b> {session.started_at}<br/>
    <b>Completed At:</b> {session.completed_at or 'N/A'}<br/>
    """
    story.append(Paragraph(meta_info, subtitle_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Summary Vignette Section
    story.append(Paragraph("1. Respondent Insights Vignette", h1_style))
    
    story.append(Paragraph("<b>First-Person Closing Narrative Vignette:</b>", h2_style))
    closing_vignette = session.vignette.summary_story or "No closing story recorded."
    story.append(Paragraph(f'"{closing_vignette}"', vignette_style))
    
    story.append(Paragraph("<b>Researcher Analytical Summary:</b>", h2_style))
    researcher_summary = session.vignette.researcher_summary or "No analytical summary generated."
    story.append(Paragraph(researcher_summary, body_style))
    
    story.append(Spacer(1, 0.2 * inch))
    
    # Sensory Analogies
    story.append(Paragraph("2. Multisensory Analogies", h1_style))
    sensory = session.sensory_analogies
    sensory_data = [
        [Paragraph("<b>Sense</b>", body_style), Paragraph("<b>Metaphor / Analogy Described</b>", body_style)],
        [Paragraph("👅 Taste", body_style), Paragraph(sensory.taste or "N/A", body_style)],
        [Paragraph("👉 Touch / Texture", body_style), Paragraph(sensory.touch or "N/A", body_style)],
        [Paragraph("👃 Smell", body_style), Paragraph(sensory.smell or "N/A", body_style)],
        [Paragraph("👂 Sound", body_style), Paragraph(sensory.sound or "N/A", body_style)]
    ]
    t_sensory = Table(sensory_data, colWidths=[2.0 * inch, 4.5 * inch])
    t_sensory.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sensory)
    
    story.append(PageBreak())
    
    # --- PAGE 2: CONSTRUCT HIERARCHY MAP DATA ---
    story.append(Paragraph("3. Attribute-Consequence-Value (ACV) Hierarchy Map", h1_style))
    story.append(Paragraph("Below are the standardized constructs extracted during the Kelly Repertory Grid and laddering process, categorized into ZMET means-end hierarchy levels.", body_style))
    
    construct_data = [
        [Paragraph("<b>ID</b>", body_style), Paragraph("<b>Construct Name</b>", body_style), Paragraph("<b>ACV Hierarchy Classification</b>", body_style)]
    ]
    for c in session.constructs:
        classification = c.type.upper()
        # Visual styling indicator based on type
        if classification == "ATTRIBUTE":
            badge_lbl = f"<font color='#0d47a1'><b>{classification}</b> (Originator)</font>"
        elif classification == "CONSEQUENCE":
            badge_lbl = f"<font color='#5d4037'><b>{classification}</b> (Connector)</font>"
        else:
            badge_lbl = f"<font color='#880e4f'><b>{classification}</b> (Destination Value)</font>"
            
        construct_data.append([
            Paragraph(c.id, body_style),
            Paragraph(c.name, body_style),
            Paragraph(badge_lbl, body_style)
        ])
        
    t_constructs = Table(construct_data, colWidths=[1.0 * inch, 3.25 * inch, 2.25 * inch])
    t_constructs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_constructs)
    
    story.append(Spacer(1, 0.25 * inch))
    
    story.append(Paragraph("4. Construct Relationship Links (Reasoning Chains)", h1_style))
    story.append(Paragraph("The connections below describe how attributes trigger consequences, which ultimately lead to terminal customer values.", body_style))
    
    relation_data = [
        [Paragraph("<b>Source</b>", body_style), Paragraph("<b>Target</b>", body_style), Paragraph("<b>Causal Explanation / Rationale</b>", body_style)]
    ]
    for r in session.relations:
        src_c = next((c for c in session.constructs if c.id == r.source), None)
        tgt_c = next((c for c in session.constructs if c.id == r.target), None)
        src_name = src_c.name if src_c else r.source
        tgt_name = tgt_c.name if tgt_c else r.target
        
        relation_data.append([
            Paragraph(f"<b>{src_name}</b>", body_style),
            Paragraph(f"<b>{tgt_name}</b>", body_style),
            Paragraph(r.explanation, body_style)
        ])
        
    t_relations = Table(relation_data, colWidths=[2.0 * inch, 2.0 * inch, 2.5 * inch])
    t_relations.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_relations)
    
    story.append(PageBreak())
    
    # --- PAGE 3: SEQUENTIAL DIALOG TRANSCRIPT ---
    story.append(Paragraph("5. Sequential Interview Dialog Log", h1_style))
    story.append(Paragraph("Below is the complete transcript of the dialogue between the AI Moderator and the Respondent.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    transcript_elements = []
    for msg in session.transcript:
        sender_lbl = "🤖 Moderator:" if msg["sender"] == "moderator" else "👤 Respondent:"
        msg_style = ParagraphStyle(
            'TranscriptMsg',
            parent=body_style,
            leftIndent=20 if msg["sender"] == "respondent" else 0,
            textColor=primary_color if msg["sender"] == "moderator" else colors.HexColor("#1e3a8a"),
            fontName='Helvetica-Bold' if msg["sender"] == "moderator" else 'Helvetica'
        )
        
        transcript_elements.append(Paragraph(f"<b>{sender_lbl}</b> {msg['text']}", msg_style))
        transcript_elements.append(Spacer(1, 4))
        
    story.append(KeepTogether(transcript_elements))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
