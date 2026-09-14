import os
import tempfile
from typing import List
from pyvis.network import Network
import networkx as nx

# Import classes from interview engine
from interview_engine import Construct, Relation

def generate_pyvis_html(constructs: List[Construct], relations: List[Relation]) -> str:
    """
    Renders the Attribute-Consequence-Value (ACV) network map.
    Returns the HTML source code of the Pyvis interactive network visualization.
    """
    # Initialize Pyvis Network
    # Using height='500px' and width='100%'
    net = Network(
        height="500px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#fafafa"
    )

    # Styling mapping
    # Attributes: Soft blue
    # Consequences: Soft warm amber
    # Values: Premium pink/rose
    style_map = {
        "attribute": {
            "color": {"background": "#E3F2FD", "border": "#1E88E5", "highlight": {"background": "#BBDEFB", "border": "#1565C0"}},
            "font": {"color": "#0D47A1", "face": "system-ui"},
            "shape": "box",
            "size": 15
        },
        "consequence": {
            "color": {"background": "#FFF8E1", "border": "#FFB300", "highlight": {"background": "#FFE082", "border": "#F57C00"}},
            "font": {"color": "#5D4037", "face": "system-ui"},
            "shape": "ellipse",
            "size": 20
        },
        "value": {
            "color": {"background": "#FCE4EC", "border": "#D81B60", "highlight": {"background": "#F8BBD0", "border": "#C2185B"}},
            "font": {"color": "#880E4F", "face": "system-ui", "size": 18, "bold": True},
            "shape": "dot",
            "size": 25
        }
    }

    # Add Nodes
    for construct in constructs:
        ctype = construct.type.lower()
        style = style_map.get(ctype, style_map["attribute"])
        
        # Build tooltip title
        title_text = f"<b>{ctype.upper()}</b><br/>"
        if construct.source_image_id:
            title_text += f"Source Image: {construct.source_image_id}<br/>"
        
        label_text = construct.name

        net.add_node(
            construct.id,
            label=label_text,
            title=title_text,
            shape=style["shape"],
            color=style["color"],
            font=style["font"],
            size=style["size"],
            borderWidth=2
        )

    # Add Edges
    for idx, relation in enumerate(relations):
        # We ensure both source and target exist in constructs list to prevent visualization crashes
        net.add_edge(
            relation.source,
            relation.target,
            title=relation.explanation,
            label="leads to",
            color="#90A4AE",
            width=2,
            arrowStrikethrough=False
        )

    # Set interactive options (physics, layout settings)
    net.set_options("""
    var options = {
      "nodes": {
        "shadow": true
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        },
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "none",
          "roundness": 0.4
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 150
        }
      }
    }
    """)

    # Save to a temporary file, read the HTML and return
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"zmet_graph_{os.getpid()}.html")
    
    try:
        net.save_graph(temp_file_path)
        with open(temp_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
