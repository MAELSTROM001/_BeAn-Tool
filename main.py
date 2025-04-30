# ======================================================================
# Beam Analysis Standalone Application
# ======================================================================
# This is a standalone version of the Beam Analysis application.
# It combines all modules into a single file for easier offline use.
#
# Sections:
# 1. Imports and Setup
# 2. ICONS (Beam and Load Icons)
# 3. Cross Sections
# 4. Load Combinations
# 5. Beam Analysis Core
# 6. Stress Analysis
# 7. Visualization
# 8. Report Generator
# 9. Main Application
# ======================================================================

# ----------------------------------------------------------------------
# 1. IMPORTS AND SETUP
# ----------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO, StringIO
import base64
import io
import sympy as sp
from sympy import symbols, integrate, solve, Function, dsolve, Eq
from scipy.integrate import cumulative_trapezoid as cumtrapz
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.units import inch
from datetime import datetime

# ----------------------------------------------------------------------
# 2. ICONS (Beam and Load Icons)
# ----------------------------------------------------------------------
# SVG Icons for different beam types
BEAM_ICONS = {
    "Simply Supported": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="30" x2="90" y2="30" stroke="black" stroke-width="4"/>
        <polygon points="10,40 0,30 10,20" fill="black"/>
        <circle cx="90" cy="30" r="5" fill="white" stroke="black" stroke-width="2"/>
    </svg>
    ''',

    "Cantilever": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="30" x2="90" y2="30" stroke="black" stroke-width="4"/>
        <rect x="0" y="15" width="10" height="30" fill="gray"/>
    </svg>
    ''',

    "Fixed-Fixed": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="30" x2="90" y2="30" stroke="black" stroke-width="4"/>
        <rect x="0" y="15" width="10" height="30" fill="gray"/>
        <rect x="90" y="15" width="10" height="30" fill="gray"/>
    </svg>
    ''',

    "Fixed-Supported": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="30" x2="90" y2="30" stroke="black" stroke-width="4"/>
        <rect x="0" y="15" width="10" height="30" fill="gray"/>
        <polygon points="90,40 80,30 90,20" fill="black"/>
    </svg>
    ''',

    "Supported-Supported": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="30" x2="90" y2="30" stroke="black" stroke-width="4"/>
        <polygon points="10,40 0,30 10,20" fill="black"/>
        <polygon points="90,40 80,30 90,20" fill="black"/>
    </svg>
    ''',

    "Overhanging": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="30" x2="100" y2="30" stroke="black" stroke-width="4"/>
        <polygon points="30,40 20,30 30,20" fill="black"/>
        <circle cx="70" cy="30" r="5" fill="white" stroke="black" stroke-width="2"/>
    </svg>
    '''
}

# SVG Icons for different load types
LOAD_ICONS = {
    "Point Load": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="40" x2="90" y2="40" stroke="black" stroke-width="4"/>
        <line x1="50" y1="10" x2="50" y2="40" stroke="red" stroke-width="2"/>
        <polygon points="50,10 45,20 55,20" fill="red"/>
    </svg>
    ''',

    "Uniform Distributed Load (UDL)": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="40" x2="90" y2="40" stroke="black" stroke-width="4"/>
        <line x1="20" y1="10" x2="20" y2="40" stroke="red" stroke-width="2"/>
        <line x1="30" y1="10" x2="30" y2="40" stroke="red" stroke-width="2"/>
        <line x1="40" y1="10" x2="40" y2="40" stroke="red" stroke-width="2"/>
        <line x1="50" y1="10" x2="50" y2="40" stroke="red" stroke-width="2"/>
        <line x1="60" y1="10" x2="60" y2="40" stroke="red" stroke-width="2"/>
        <line x1="70" y1="10" x2="70" y2="40" stroke="red" stroke-width="2"/>
        <line x1="80" y1="10" x2="80" y2="40" stroke="red" stroke-width="2"/>
        <line x1="20" y1="10" x2="80" y2="10" stroke="red" stroke-width="2"/>
    </svg>
    ''',

    "Linearly Varying Load": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="40" x2="90" y2="40" stroke="black" stroke-width="4"/>
        <line x1="20" y1="30" x2="20" y2="40" stroke="red" stroke-width="2"/>
        <line x1="30" y1="25" x2="30" y2="40" stroke="red" stroke-width="2"/>
        <line x1="40" y1="20" x2="40" y2="40" stroke="red" stroke-width="2"/>
        <line x1="50" y1="15" x2="50" y2="40" stroke="red" stroke-width="2"/>
        <line x1="60" y1="10" x2="60" y2="40" stroke="red" stroke-width="2"/>
        <line x1="70" y1="10" x2="70" y2="40" stroke="red" stroke-width="2"/>
        <line x1="80" y1="10" x2="80" y2="40" stroke="red" stroke-width="2"/>
        <line x1="20" y1="30" x2="80" y2="10" stroke="red" stroke-width="2"/>
    </svg>
    ''',

    "Moment": '''
    <svg width="100" height="60" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="40" x2="90" y2="40" stroke="black" stroke-width="4"/>
        <path d="M 50,25 A 15,15 0 0 1 35,40" stroke="red" stroke-width="2" fill="none"/>
        <polygon points="50,25 45,15 55,20" fill="red"/>
    </svg>
    '''
}

# ----------------------------------------------------------------------
# 3. CROSS SECTIONS
# ----------------------------------------------------------------------
# Standard cross-section library
# I-Sections (Wide Flange/W-Sections) - Dimensions in meters
I_SECTIONS = {
    "W8x31": {
        "height": 0.2032,  # 8 inches
        "width": 0.2032,  # 8 inches
        "web_thickness": 0.0079,
        "flange_thickness": 0.0127,
        "I_xx": 4.16e-5,  # m⁴
        "I_yy": 1.41e-5,  # m⁴
        "area": 0.00589,  # m²
        "material": "Steel"
    },
    "W10x45": {
        "height": 0.254,  # 10 inches
        "width": 0.2032,  # 8 inches
        "web_thickness": 0.0095,
        "flange_thickness": 0.0159,
        "I_xx": 8.32e-5,  # m⁴
        "I_yy": 2.83e-5,  # m⁴
        "area": 0.00854,  # m²
        "material": "Steel"
    },
    "W12x65": {
        "height": 0.3048,  # 12 inches
        "width": 0.3048,  # 12 inches
        "web_thickness": 0.0106,
        "flange_thickness": 0.0189,
        "I_xx": 1.58e-4,  # m⁴
        "I_yy": 5.37e-5,  # m⁴
        "area": 0.01234,  # m²
        "material": "Steel"
    }
}

# Rectangular Sections
RECTANGULAR_SECTIONS = {
    "Rectangular 100x200": {
        "height": 0.2,  # 200 mm
        "width": 0.1,  # 100 mm
        "I_xx": 6.67e-5,  # m⁴ (b*h³/12)
        "I_yy": 1.67e-5,  # m⁴ (h*b³/12)
        "area": 0.02,  # m²
        "material": "Timber"
    },
    "Rectangular 150x300": {
        "height": 0.3,  # 300 mm
        "width": 0.15,  # 150 mm
        "I_xx": 3.38e-4,  # m⁴
        "I_yy": 4.22e-5,  # m⁴
        "area": 0.045,  # m²
        "material": "Concrete"
    },
    "Rectangular 200x400": {
        "height": 0.4,  # 400 mm
        "width": 0.2,  # 200 mm
        "I_xx": 1.07e-3,  # m⁴
        "I_yy": 1.33e-4,  # m⁴
        "area": 0.08,  # m²
        "material": "Concrete"
    }
}

# Circular Sections
CIRCULAR_SECTIONS = {
    "Circular Diameter 100": {
        "diameter": 0.1,  # 100 mm
        "I": 4.91e-6,  # m⁴ (πd⁴/64)
        "area": 0.00785,  # m² (πd²/4)
        "material": "Steel"
    },
    "Circular Diameter 150": {
        "diameter": 0.15,  # 150 mm
        "I": 2.49e-5,  # m⁴
        "area": 0.01767,  # m²
        "material": "Steel"
    },
    "Circular Diameter 200": {
        "diameter": 0.2,  # 200 mm
        "I": 7.85e-5,  # m⁴
        "area": 0.0314,  # m²
        "material": "Aluminum"
    }
}

# Collect all sections in one dictionary
ALL_SECTIONS = {}
ALL_SECTIONS.update(I_SECTIONS)
ALL_SECTIONS.update(RECTANGULAR_SECTIONS)
ALL_SECTIONS.update(CIRCULAR_SECTIONS)


def get_section_names_by_category():
    """
    Return a dictionary of section names organized by category.
    """
    return {
        "I-Sections": list(I_SECTIONS.keys()),
        "Rectangular Sections": list(RECTANGULAR_SECTIONS.keys()),
        "Circular Sections": list(CIRCULAR_SECTIONS.keys())
    }


def get_section_properties(section_name):
    """
    Get the properties of a specific section.

    Parameters:
    -----------
    section_name : str
        Name of the section

    Returns:
    --------
    dict
        Dictionary containing section properties
    """
    return ALL_SECTIONS.get(section_name, None)


def get_moment_of_inertia(section_name, axis="xx"):
    """
    Get the moment of inertia for a specific section and axis.

    Parameters:
    -----------
    section_name : str
        Name of the section
    axis : str
        Axis for the moment of inertia ("xx" or "yy")

    Returns:
    --------
    float
        Moment of inertia in m⁴
    """
    section = get_section_properties(section_name)
    if not section:
        return None

    if "I" in section:  # Circular section
        return section["I"]

    if axis == "xx" and "I_xx" in section:
        return section["I_xx"]
    elif axis == "yy" and "I_yy" in section:
        return section["I_yy"]

    return None


# ----------------------------------------------------------------------
# 4. LOAD COMBINATIONS
# ----------------------------------------------------------------------
# Define load combination factors based on different design codes

# ASCE 7 Load Combinations (U.S. Building Code)
ASCE7_COMBINATIONS = {
    "1.4D": {
        "description": "Strength Design - Dead Load Only",
        "factors": {
            "Dead Load": 1.4
        }
    },
    "1.2D + 1.6L": {
        "description": "Strength Design - Primary Gravity Combination",
        "factors": {
            "Dead Load": 1.2,
            "Live Load": 1.6
        }
    },
    "1.2D + 1.0W + 1.0L": {
        "description": "Strength Design - Wind Combination",
        "factors": {
            "Dead Load": 1.2,
            "Live Load": 1.0,
            "Wind Load": 1.0
        }
    },
    "0.9D + 1.0W": {
        "description": "Strength Design - Wind Uplift Combination",
        "factors": {
            "Dead Load": 0.9,
            "Wind Load": 1.0
        }
    },
    "1.2D + 1.0E + 1.0L": {
        "description": "Strength Design - Seismic Combination",
        "factors": {
            "Dead Load": 1.2,
            "Live Load": 1.0,
            "Seismic Load": 1.0
        }
    },
    "0.9D + 1.0E": {
        "description": "Strength Design - Seismic Uplift Combination",
        "factors": {
            "Dead Load": 0.9,
            "Seismic Load": 1.0
        }
    }
}

# Eurocode Load Combinations (European Standard)
EUROCODE_COMBINATIONS = {
    "1.35G": {
        "description": "Ultimate Limit State - Permanent Actions Only",
        "factors": {
            "Dead Load": 1.35
        }
    },
    "1.35G + 1.5Q": {
        "description": "Ultimate Limit State - Primary Gravity Combination",
        "factors": {
            "Dead Load": 1.35,
            "Live Load": 1.5
        }
    },
    "1.35G + 1.5Q + 0.9W": {
        "description": "Ultimate Limit State - Primary with Wind",
        "factors": {
            "Dead Load": 1.35,
            "Live Load": 1.5,
            "Wind Load": 0.9
        }
    },
    "1.35G + 1.5W + 1.05Q": {
        "description": "Ultimate Limit State - Wind Dominant",
        "factors": {
            "Dead Load": 1.35,
            "Live Load": 1.05,
            "Wind Load": 1.5
        }
    },
    "1.0G + 1.0Q": {
        "description": "Serviceability Limit State - Characteristic Combination",
        "factors": {
            "Dead Load": 1.0,
            "Live Load": 1.0
        }
    }
}

# Australian/New Zealand Standard (AS/NZS 1170)
ASNZS_COMBINATIONS = {
    "1.35G": {
        "description": "Ultimate Limit State - Permanent Action Only",
        "factors": {
            "Dead Load": 1.35
        }
    },
    "1.2G + 1.5Q": {
        "description": "Ultimate Limit State - Primary Gravity Combination",
        "factors": {
            "Dead Load": 1.2,
            "Live Load": 1.5
        }
    },
    "1.2G + 0.4Q + 1.0W": {
        "description": "Ultimate Limit State - Wind Dominant",
        "factors": {
            "Dead Load": 1.2,
            "Live Load": 0.4,
            "Wind Load": 1.0
        }
    },
    "0.9G + 1.0W": {
        "description": "Ultimate Limit State - Wind Uplift",
        "factors": {
            "Dead Load": 0.9,
            "Wind Load": 1.0
        }
    },
    "G + 0.7Q": {
        "description": "Serviceability Limit State - Long-term",
        "factors": {
            "Dead Load": 1.0,
            "Live Load": 0.7
        }
    }
}

# Dictionary of supported design codes
DESIGN_CODES = {
    "ASCE 7 (US)": ASCE7_COMBINATIONS,
    "Eurocode (EU)": EUROCODE_COMBINATIONS,
    "AS/NZS 1170 (AUS/NZ)": ASNZS_COMBINATIONS
}

# Load types for categorization
LOAD_TYPES = ["Dead Load", "Live Load", "Wind Load", "Seismic Load"]


def get_design_codes():
    """
    Get a list of supported design codes.

    Returns:
    --------
    list
        List of supported design code names
    """
    return list(DESIGN_CODES.keys())


def get_combinations_for_code(design_code):
    """
    Get all load combinations for a specific design code.

    Parameters:
    -----------
    design_code : str
        Name of the design code

    Returns:
    --------
    dict
        Dictionary of load combinations for the specified code
    """
    if design_code in DESIGN_CODES:
        return DESIGN_CODES[design_code]
    return {}


def apply_load_combination(loads, load_types, combination_factors):
    """
    Apply load combination factors to a set of loads.

    Parameters:
    -----------
    loads : list
        List of load dictionaries
    load_types : list
        List of load type designations (same length as loads)
    combination_factors : dict
        Dictionary of load factors keyed by load type

    Returns:
    --------
    list
        New list of load dictionaries with factors applied
    """
    combined_loads = []

    for i, load in enumerate(loads):
        load_type = load_types[i]

        # Skip loads whose type is not in the combination
        if load_type not in combination_factors:
            continue

        # Get the factor for this load type
        factor = combination_factors[load_type]

        # Create a copy of the load with the factor applied
        combined_load = load.copy()

        # Apply the factor to the magnitude fields
        if "magnitude" in combined_load:
            combined_load["magnitude"] *= factor
        if "start_magnitude" in combined_load:
            combined_load["start_magnitude"] *= factor
        if "end_magnitude" in combined_load:
            combined_load["end_magnitude"] *= factor

        combined_loads.append(combined_load)

    return combined_loads

# ----------------------------------------------------------------------
# 5. BEAM ANALYSIS CORE
# ----------------------------------------------------------------------
def get_beam_equations(beam_type, beam_length, loads):
    """
    Generate symbolic equations for the beam based on its type and loads.

    Parameters:
    -----------
    beam_type : str
        Type of beam (e.g., "Simply Supported", "Cantilever")
    beam_length : float
        Length of the beam in meters
    loads : list
        List of load dictionaries containing load information

    Returns:
    --------
    dict
        Dictionary containing symbolic equations in LaTeX format
    """
    # Define symbolic variables
    x, L, E, I = sp.symbols('x L E I')

    # Initialize equation components
    shear_expr = 0
    moment_expr = 0

    # Define boundary conditions based on beam type
    if beam_type == "Simply Supported":
        # Boundary conditions for simply supported beam:
        # v(0) = 0, v(L) = 0 (zero deflection at supports)
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': L, 'condition': 'deflection', 'value': 0}
        ]
    elif beam_type == "Cantilever":
        # Boundary conditions for cantilever beam:
        # v(0) = 0, v'(0) = 0 (zero deflection and slope at fixed end)
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': 0, 'condition': 'slope', 'value': 0}
        ]
    elif beam_type == "Fixed-Fixed":
        # Boundary conditions for fixed-fixed beam:
        # v(0) = 0, v'(0) = 0, v(L) = 0, v'(L) = 0
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': 0, 'condition': 'slope', 'value': 0},
            {'point': L, 'condition': 'deflection', 'value': 0},
            {'point': L, 'condition': 'slope', 'value': 0}
        ]
    elif beam_type == "Fixed-Supported":
        # Boundary conditions for fixed-supported beam:
        # v(0) = 0, v'(0) = 0, v(L) = 0
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': 0, 'condition': 'slope', 'value': 0},
            {'point': L, 'condition': 'deflection', 'value': 0}
        ]
    elif beam_type == "Supported-Supported":
        # Same as simply supported
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': L, 'condition': 'deflection', 'value': 0}
        ]
    elif beam_type == "Overhanging":
        # For simplicity, treat as simply supported with specific support locations
        # Support locations would need to be defined for actual implementation
        boundary_conditions = [
            {'point': 0.2 * L, 'condition': 'deflection', 'value': 0},
            {'point': 0.8 * L, 'condition': 'deflection', 'value': 0}
        ]
    else:
        # Default to simply supported
        boundary_conditions = [
            {'point': 0, 'condition': 'deflection', 'value': 0},
            {'point': L, 'condition': 'deflection', 'value': 0}
        ]

    # Add load expressions
    for load in loads:
        load_type = load["type"]

        if load_type == "Point Load":
            position = load["position"]
            magnitude = load["magnitude"]

            # Use Heaviside step function for point load
            pos_sym = position * L / beam_length  # Convert position to symbolic
            shear_expr += -magnitude * sp.Heaviside(x - pos_sym)
            moment_expr += -magnitude * sp.Heaviside(x - pos_sym) * (x - pos_sym)

        elif load_type == "UDL":
            start = load["start"]
            end = load["end"]
            magnitude = load["magnitude"]

            # Convert positions to symbolic
            start_sym = start * L / beam_length
            end_sym = end * L / beam_length

            # Distributed load expression
            shear_expr += -magnitude * (sp.Heaviside(x - start_sym) - sp.Heaviside(x - end_sym))
            moment_expr += -magnitude * ((x - start_sym) * sp.Heaviside(x - start_sym) -
                                         (x - end_sym) * sp.Heaviside(x - end_sym))

        elif load_type == "Linearly Varying":
            start = load["start"]
            end = load["end"]
            start_magnitude = load["start_magnitude"]
            end_magnitude = load["end_magnitude"]

            # Convert positions to symbolic
            start_sym = start * L / beam_length
            end_sym = end * L / beam_length

            # Linearly varying load expression
            # For simplicity, treat as equivalent uniform loads in this symbolic representation
            avg_magnitude = (start_magnitude + end_magnitude) / 2
            shear_expr += -avg_magnitude * (sp.Heaviside(x - start_sym) - sp.Heaviside(x - end_sym))
            moment_expr += -avg_magnitude * ((x - start_sym) * sp.Heaviside(x - start_sym) -
                                             (x - end_sym) * sp.Heaviside(x - end_sym))

        elif load_type == "Moment":
            position = load["position"]
            magnitude = load["magnitude"]

            # Convert position to symbolic
            pos_sym = position * L / beam_length

            # Moment at a point affects curvature directly
            moment_expr += magnitude * sp.DiracDelta(x - pos_sym)

    # Convert expressions to LaTeX format
    shear_latex = sp.latex(shear_expr)
    moment_latex = sp.latex(moment_expr)

    # Simplify for display
    shear_latex = shear_latex.replace("\\operatorname{Heaviside}", "H")
    moment_latex = moment_latex.replace("\\operatorname{Heaviside}", "H")
    moment_latex = moment_latex.replace("\\operatorname{DiracDelta}", "\\delta")

    # Return equations in LaTeX format
    return {
        "Shear Force, V(x)": shear_latex,
        "Bending Moment, M(x)": moment_latex,
        "Slope, θ(x)": "\\int \\frac{M(x)}{EI} dx + C_1",
        "Deflection, v(x)": "\\int \\int \\frac{M(x)}{EI} dx dx + C_1 x + C_2"
    }


def calculate_beam_response(beam_type, beam_length, loads, elastic_modulus, moment_of_inertia):
    """
    Calculate the response of a beam under given loads.

    Parameters:
    -----------
    beam_type : str
        Type of beam (e.g., "Simply Supported", "Cantilever")
    beam_length : float
        Length of the beam in meters
    loads : list
        List of load dictionaries containing load information
    elastic_modulus : float
        Elastic modulus of the beam material in N/m²
    moment_of_inertia : float
        Moment of inertia of the beam cross-section in m⁴

    Returns:
    --------
    dict
        Dictionary containing the calculated beam response
    """
    # Create an array of x positions along the beam
    num_points = 1001  # Increase resolution for smoother plots
    x = np.linspace(0, beam_length, num_points)

    # Initialize arrays for shear force, bending moment, slope, and deflection
    shear_force = np.zeros_like(x)
    bending_moment = np.zeros_like(x)

    # Add effect of each load
    for load in loads:
        load_type = load["type"]

        if load_type == "Point Load":
            position = load["position"]
            magnitude = load["magnitude"]

            # Shear force
            shear_force += -magnitude * (x >= position)

            # Bending moment
            bending_moment += -magnitude * np.maximum(0, x - position)

        elif load_type == "UDL":
            start = load["start"]
            end = load["end"]
            magnitude = load["magnitude"]

            # Shear force
            shear_force += -magnitude * (
                    (x >= start) * (x - start) -
                    (x >= end) * (x - end)
            )

            # Bending moment
            bending_moment += -magnitude * (
                    0.5 * (x >= start) * ((x - start) ** 2) -
                    0.5 * (x >= end) * ((x - end) ** 2)
            )

        elif load_type == "Linearly Varying":
            start = load["start"]
            end = load["end"]
            start_magnitude = load["start_magnitude"]
            end_magnitude = load["end_magnitude"]

            # Calculate slope of load intensity
            if end > start:
                load_slope = (end_magnitude - start_magnitude) / (end - start)
            else:
                load_slope = 0

            # For x between start and end
            mask = (x >= start) & (x <= end)

            # Coefficients for each segment
            a = load_slope / 2
            b = start_magnitude - load_slope * start

            # Shear force and bending moment calculations
            for i, xi in enumerate(x):
                if start <= xi <= end:
                    # Distance from start of load
                    d = xi - start
                    # Shear force formula for linearly varying load
                    F_left = a * d ** 2 + b * d
                    shear_force[i] += -F_left

                    # Bending moment formula
                    M_left = (a * d ** 3 / 3) + (b * d ** 2 / 2)
                    bending_moment[i] += -M_left
                elif xi > end:
                    # Total shear force due to entire load
                    F_total = (a * (end - start) ** 2) + (b * (end - start))
                    shear_force[i] += -F_total

                    # Distance from start to end of load
                    d_total = end - start
                    # Determine centroid of load
                    if a == 0:  # Uniform load
                        centroid = d_total / 2
                    else:  # Linearly varying load
                        # Formula for centroid of a trapezoid
                        h1 = start_magnitude
                        h2 = end_magnitude
                        centroid = d_total * (h1 + 2 * h2) / (3 * (h1 + h2)) if (h1 + h2) > 0 else d_total / 2

                    # Moment due to total force at centroid
                    M_total = F_total * (xi - (start + centroid))
                    bending_moment[i] += -M_total

        elif load_type == "Moment":
            position = load["position"]
            magnitude = load["magnitude"]

            # Bending moment due to a point moment
            bending_moment += magnitude * (x >= position)

    # Add reactions based on beam type and boundary conditions
    reactions = calculate_reactions(beam_type, beam_length, loads)

    for reaction_name, reaction_value in reactions.items():
        if reaction_name == "R_A":  # Support at x=0
            shear_force += reaction_value * (x >= 0)
            bending_moment += reaction_value * x * (x >= 0)
        elif reaction_name == "R_B":  # Support at x=L
            shear_force += reaction_value * (x >= beam_length)
            bending_moment += reaction_value * (x - beam_length) * (x >= beam_length)
        elif reaction_name == "M_A":  # Moment at x=0
            bending_moment += -reaction_value * (x >= 0)
        elif reaction_name == "M_B":  # Moment at x=L
            bending_moment += -reaction_value * (x >= beam_length)

    # Calculate deflection using numerical integration
    # EI * d²v/dx² = M(x)
    # First integrate to get slope (θ = dv/dx)
    # Then integrate again to get deflection

    # Factor for numerical integration
    EI = elastic_modulus * moment_of_inertia

    # First integration - Slope
    slope = np.zeros_like(x)
    slope = cumtrapz(bending_moment / EI, x, initial=0)

    # Apply boundary conditions to find integration constants
    # Function to find constants given the known values at specific points
    def find_constants(a_idx, b_idx):
        if beam_type == "Simply Supported":
            # For simply supported: v(0) = 0, v(L) = 0
            A = np.array([[1, 0], [1, x[b_idx]]])
            b = np.array([0, -cumtrapz(slope, x, initial=0)[b_idx]])
            return np.linalg.solve(A, b)

        elif beam_type == "Cantilever":
            # For cantilever: v(0) = 0, v'(0) = 0
            return [0, -slope[0]]

        elif beam_type == "Fixed-Fixed":
            # For fixed-fixed: v(0) = 0, v'(0) = 0, v(L) = 0, v'(L) = 0
            # We need to adjust the entire slope curve to meet these conditions
            c1 = -slope[0]  # offset to make v'(0) = 0
            c2 = 0  # offset to make v(0) = 0

            # Adjust slope with c1
            adjusted_slope = slope + c1

            # Preliminary deflection with C1 applied
            prelim_deflection = cumtrapz(adjusted_slope, x, initial=0)

            # Calculate remaining correction needed at x=L
            correction = -prelim_deflection[b_idx] / x[b_idx]

            # Final constants
            return [c1 + correction, c2]

        elif beam_type == "Fixed-Supported":
            # For fixed-supported: v(0) = 0, v'(0) = 0, v(L) = 0
            c1 = -slope[0]  # offset to make v'(0) = 0

            # Adjust slope with c1
            adjusted_slope = slope + c1

            # Calculate deflection
            prelim_deflection = cumtrapz(adjusted_slope, x, initial=0)

            # Find C2 to make v(L) = 0
            c2 = -prelim_deflection[b_idx]

            return [c1, c2]

        elif beam_type == "Supported-Supported":
            # Same as simply supported
            A = np.array([[1, 0], [1, x[b_idx]]])
            b = np.array([0, -cumtrapz(slope, x, initial=0)[b_idx]])
            return np.linalg.solve(A, b)

        elif beam_type == "Overhanging":
            # Treat as simply supported with support locations at 20% and 80% of span
            # for demonstration purposes
            a_idx = int(0.2 * len(x))
            b_idx = int(0.8 * len(x))

            A = np.array([[1, x[a_idx]], [1, x[b_idx]]])
            deflection_prelim = cumtrapz(slope, x, initial=0)
            b = np.array([-deflection_prelim[a_idx], -deflection_prelim[b_idx]])

            try:
                constants = np.linalg.solve(A, b)
                return constants
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                return [0, 0]

        else:
            # Default to simply supported
            A = np.array([[1, 0], [1, x[b_idx]]])
            b = np.array([0, -cumtrapz(slope, x, initial=0)[b_idx]])
            return np.linalg.solve(A, b)

    # Find constants
    a_idx = 0  # First point (x=0)
    b_idx = -1  # Last point (x=L)

    constants = find_constants(a_idx, b_idx)
    C1, C2 = constants

    # Second integration and apply constants - Deflection
    slope = slope + C1
    deflection = cumtrapz(slope, x, initial=0) + C2

    # Convert reactions to a dictionary
    reactions_dict = reactions

    return {
        "x": x,
        "shear_force": shear_force,
        "bending_moment": bending_moment,
        "slope": slope,
        "deflection": deflection,
        "reactions": reactions_dict
    }


def calculate_reactions(beam_type, beam_length, loads):
    """
    Calculate the support reactions for a beam under given loads.

    Parameters:
    -----------
    beam_type : str
        Type of beam (e.g., "Simply Supported", "Cantilever")
    beam_length : float
        Length of the beam in meters
    loads : list
        List of load dictionaries containing load information

    Returns:
    --------
    dict
        Dictionary containing the calculated reactions
    """
    # Initialize reaction forces and moments
    reactions = {}

    if beam_type == "Simply Supported":
        # Two unknown reaction forces: R_A and R_B
        R_A = 0
        R_B = 0

        # Calculate reactions for each load
        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions
                R_B += magnitude * position / beam_length
                R_A += magnitude * (1 - position / beam_length)

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = magnitude * (end - start)
                # Center of force
                center = (start + end) / 2

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Linearly Varying":
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions (counter-clockwise positive)
                R_B += -magnitude / beam_length
                R_A += magnitude / beam_length

        reactions = {
            "R_A": R_A,
            "R_B": R_B
        }

    elif beam_type == "Cantilever":
        # One reaction force and one reaction moment at the fixed end
        R_A = 0
        M_A = 0

        # Calculate reactions for each load
        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions
                R_A += magnitude
                M_A += magnitude * position

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = magnitude * (end - start)
                # Center of force
                center = (start + end) / 2

                # Contribution to reactions
                R_A += total_force
                M_A += total_force * center

            elif load_type == "Linearly Varying":
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                # Contribution to reactions
                R_A += total_force
                M_A += total_force * center

            elif load_type == "Moment":
                magnitude = load["magnitude"]

                # Direct contribution to moment reaction
                M_A += magnitude

        reactions = {
            "R_A": R_A,
            "M_A": M_A
        }

    elif beam_type == "Fixed-Fixed":
        # Two reaction forces and two reaction moments
        # This requires solving a statically indeterminate structure
        # For simplicity, we'll use a basic approximation

        # Rough approximation based on superposition
        # For each load, calculate fixed-end moments and then forces
        R_A = 0
        R_B = 0
        M_A = 0
        M_B = 0

        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]
                a = position
                b = beam_length - position

                # Fixed-end moments (FEM)
                M_A += -magnitude * a * b ** 2 / beam_length ** 2
                M_B += magnitude * a ** 2 * b / beam_length ** 2

                # Reaction forces
                R_A += magnitude * b ** 2 * (beam_length + 2 * a) / beam_length ** 3
                R_B += magnitude * a ** 2 * (beam_length + 2 * b) / beam_length ** 3

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Treat as a point load for approximation
                length = end - start
                center = (start + end) / 2
                total_force = magnitude * length

                a = center
                b = beam_length - center

                # Fixed-end moments (FEM) approximation
                M_A += -total_force * a * b ** 2 / beam_length ** 2
                M_B += total_force * a ** 2 * b / beam_length ** 2

                # Reaction forces approximation
                R_A += total_force * b ** 2 * (beam_length + 2 * a) / beam_length ** 3
                R_B += total_force * a ** 2 * (beam_length + 2 * b) / beam_length ** 3

            elif load_type == "Linearly Varying":
                # Approximate as a point load at the centroid
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                a = center
                b = beam_length - center

                # Fixed-end moments (FEM) approximation
                M_A += -total_force * a * b ** 2 / beam_length ** 2
                M_B += total_force * a ** 2 * b / beam_length ** 2

                # Reaction forces approximation
                R_A += total_force * b ** 2 * (beam_length + 2 * a) / beam_length ** 3
                R_B += total_force * a ** 2 * (beam_length + 2 * b) / beam_length ** 3

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]
                a = position
                b = beam_length - position

                # Fixed-end moments (FEM) for a point moment
                M_A += -magnitude * b / beam_length
                M_B += -magnitude * a / beam_length

                # Reaction forces
                R_A += magnitude * 6 * a * b / beam_length ** 3
                R_B += -magnitude * 6 * a * b / beam_length ** 3

        reactions = {
            "R_A": R_A,
            "R_B": R_B,
            "M_A": M_A,
            "M_B": M_B
        }

    elif beam_type == "Fixed-Supported":
        # One fixed end, one simple support
        # Two reaction forces and one reaction moment
        R_A = 0
        R_B = 0
        M_A = 0

        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                # Approximations based on superposition
                R_B += magnitude * (position ** 2) * (3 * beam_length - 2 * position) / beam_length ** 3
                R_A += magnitude - R_B
                M_A += -magnitude * position * (beam_length - position) ** 2 / beam_length ** 2

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Treat as a point load for approximation
                length = end - start
                center = (start + end) / 2
                total_force = magnitude * length

                # Approximations
                R_B += total_force * (center ** 2) * (3 * beam_length - 2 * center) / beam_length ** 3
                R_A += total_force - R_B
                M_A += -total_force * center * (beam_length - center) ** 2 / beam_length ** 2

            elif load_type == "Linearly Varying":
                # Approximate as a point load at the centroid
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                # Approximations
                R_B += total_force * (center ** 2) * (3 * beam_length - 2 * center) / beam_length ** 3
                R_A += total_force - R_B
                M_A += -total_force * center * (beam_length - center) ** 2 / beam_length ** 2

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]

                # Approximations
                R_B += -magnitude * 6 * position * (beam_length - position) / beam_length ** 3
                R_A += -R_B
                M_A += -magnitude * (beam_length - position) ** 2 / beam_length ** 2

        reactions = {
            "R_A": R_A,
            "R_B": R_B,
            "M_A": M_A
        }

    elif beam_type == "Supported-Supported":
        # Same as simply supported
        R_A = 0
        R_B = 0

        # Calculate reactions for each load
        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions
                R_B += magnitude * position / beam_length
                R_A += magnitude * (1 - position / beam_length)

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = magnitude * (end - start)
                # Center of force
                center = (start + end) / 2

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Linearly Varying":
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions (counter-clockwise positive)
                R_B += -magnitude / beam_length
                R_A += magnitude / beam_length

        reactions = {
            "R_A": R_A,
            "R_B": R_B
        }

    elif beam_type == "Overhanging":
        # For simplicity, treat as simply supported with supports at 20% and 80% of span
        support_a = 0.2 * beam_length
        support_b = 0.8 * beam_length

        R_A = 0
        R_B = 0
        effective_length = support_b - support_a

        # Calculate reactions for each load
        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                if position <= support_a:
                    # Load is to the left of first support (overhang)
                    moment_a = magnitude * (support_a - position)
                    R_B = moment_a / effective_length
                    R_A = -R_B + magnitude
                elif position >= support_b:
                    # Load is to the right of second support (overhang)
                    moment_b = magnitude * (position - support_b)
                    R_A = moment_b / effective_length
                    R_B = -R_A + magnitude
                else:
                    # Load is between supports
                    R_B = magnitude * (position - support_a) / effective_length
                    R_A = magnitude - R_B

            elif load_type in ["UDL", "Linearly Varying"]:
                # Approximate as point loads
                if load_type == "UDL":
                    start = load["start"]
                    end = load["end"]
                    magnitude = load["magnitude"]

                    # Total force and center
                    total_force = magnitude * (end - start)
                    center = (start + end) / 2
                else:  # Linearly Varying
                    start = load["start"]
                    end = load["end"]
                    start_magnitude = load["start_magnitude"]
                    end_magnitude = load["end_magnitude"]

                    # Total force
                    total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                    # Center (centroid of trapezoid)
                    if start_magnitude == end_magnitude:  # Rectangle
                        center = (start + end) / 2
                    else:  # Trapezoid
                        center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                    3 * (end_magnitude + start_magnitude))

                # Apply as a point load at the center
                if center <= support_a:
                    # Force is to the left of first support
                    moment_a = total_force * (support_a - center)
                    R_B = moment_a / effective_length
                    R_A = -R_B + total_force
                elif center >= support_b:
                    # Force is to the right of second support
                    moment_b = total_force * (center - support_b)
                    R_A = moment_b / effective_length
                    R_B = -R_A + total_force
                else:
                    # Force is between supports
                    R_B = total_force * (center - support_a) / effective_length
                    R_A = total_force - R_B

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]

                if position <= support_a:
                    # Moment is to the left of first support
                    R_B = magnitude / effective_length
                    R_A = -R_B
                elif position >= support_b:
                    # Moment is to the right of second support
                    R_A = -magnitude / effective_length
                    R_B = -R_A
                else:
                    # Moment is between supports
                    R_B = -magnitude * (support_b - position) / effective_length ** 2
                    R_A = magnitude * (position - support_a) / effective_length ** 2

        reactions = {
            "R_A": R_A,
            "R_B": R_B
        }

    else:
        # Default to simply supported
        R_A = 0
        R_B = 0

        # Calculate reactions for each load
        for load in loads:
            load_type = load["type"]

            if load_type == "Point Load":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions
                R_B += magnitude * position / beam_length
                R_A += magnitude * (1 - position / beam_length)

            elif load_type == "UDL":
                magnitude = load["magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = magnitude * (end - start)
                # Center of force
                center = (start + end) / 2

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Linearly Varying":
                start_magnitude = load["start_magnitude"]
                end_magnitude = load["end_magnitude"]
                start = load["start"]
                end = load["end"]

                # Total force
                total_force = (start_magnitude + end_magnitude) * (end - start) / 2

                # Center of force (centroid of trapezoid)
                if start_magnitude == end_magnitude:  # Rectangle
                    center = (start + end) / 2
                else:  # Trapezoid
                    center = start + (end - start) * (2 * end_magnitude + start_magnitude) / (
                                3 * (end_magnitude + start_magnitude))

                # Contribution to reactions
                R_B += total_force * center / beam_length
                R_A += total_force * (1 - center / beam_length)

            elif load_type == "Moment":
                magnitude = load["magnitude"]
                position = load["position"]

                # Contribution to reactions (counter-clockwise positive)
                R_B += -magnitude / beam_length
                R_A += magnitude / beam_length

        reactions = {
            "R_A": R_A,
            "R_B": R_B
        }

    return reactions

# ----------------------------------------------------------------------
# 6. STRESS ANALYSIS
# ----------------------------------------------------------------------
# Material yield strengths in Pa (N/m²)
MATERIAL_YIELD_STRENGTHS = {
    "Steel": 250e6,  # 250 MPa (mild structural steel)
    "Aluminum": 180e6,  # 180 MPa (6061-T6 aluminum)
    "Timber": 40e6,  # 40 MPa (parallel to grain)
    "Concrete": 30e6  # 30 MPa (compression only)
}


def calculate_normal_stress(moment, distance_from_neutral_axis, moment_of_inertia):
    """
    Calculate normal stress due to bending at a specific point on the cross-section.

    Parameters:
    -----------
    moment : float or numpy.ndarray
        Bending moment value(s) in N·m
    distance_from_neutral_axis : float
        Distance from neutral axis in meters (positive for tension, negative for compression)
    moment_of_inertia : float
        Moment of inertia of the cross-section in m⁴

    Returns:
    --------
    float or numpy.ndarray
        Normal stress value(s) in Pa (N/m²)
    """
    # Normal stress due to bending: sigma = M * y / I
    stress = moment * distance_from_neutral_axis / moment_of_inertia
    return stress


def calculate_shear_stress(shear_force, first_moment_of_area, width, moment_of_inertia):
    """
    Calculate shear stress at a specific point on the cross-section.

    Parameters:
    -----------
    shear_force : float or numpy.ndarray
        Shear force value(s) in N
    first_moment_of_area : float
        First moment of area (Q = A * y_bar) in m³
    width : float
        Width of the section at the point of interest in m
    moment_of_inertia : float
        Moment of inertia of the cross-section in m⁴

    Returns:
    --------
    float or numpy.ndarray
        Shear stress value(s) in Pa (N/m²)
    """
    # Shear stress formula: tau = (V * Q) / (I * t)
    stress = (shear_force * first_moment_of_area) / (moment_of_inertia * width)
    return stress


def calculate_first_moment_rectangular(height, width, y_location):
    """
    Calculate first moment of area for a rectangular section at a specific y-location.

    Parameters:
    -----------
    height : float
        Height of the rectangular section in m
    width : float
        Width of the rectangular section in m
    y_location : float
        Y-coordinate where to calculate the first moment (from bottom of section) in m

    Returns:
    --------
    float
        First moment of area in m³
    """
    # Neutral axis is at height/2
    neutral_axis = height / 2

    # Distance from neutral axis to the location of interest
    y_from_na = neutral_axis - y_location

    # Calculate portion of the section above the y_location
    if y_location >= height:
        # Location is above the section
        return 0
    elif y_location <= 0:
        # Location is below the section, take the entire section
        return width * height * neutral_axis
    else:
        # Location is within the section
        partial_height = height - y_location
        partial_area = width * partial_height
        y_centroid = y_location + partial_height / 2
        return partial_area * (y_centroid - neutral_axis)


def calculate_stress_distribution(results, section_properties, material):
    """
    Calculate stress distributions for a beam with given results and section properties.

    Parameters:
    -----------
    results : dict
        Results dictionary from beam analysis
    section_properties : dict
        Dictionary containing section properties
    material : str
        Material name (e.g., 'Steel', 'Concrete')

    Returns:
    --------
    dict
        Dictionary containing stress distributions and safety factors
    """
    # Extract results and properties
    x = results["x"]
    shear_force = results["shear_force"]
    bending_moment = results["bending_moment"]

    # Extract section properties
    moment_of_inertia = section_properties.get("I_xx", 0)
    if moment_of_inertia == 0:
        moment_of_inertia = section_properties.get("I", 0)  # For circular sections

    height = section_properties.get("height", 0)
    width = section_properties.get("width", section_properties.get("diameter", 0))
    if "diameter" in section_properties:
        # Circular section
        height = width

    # Calculate extreme fiber distances
    y_top = height / 2
    y_bottom = -height / 2

    # Calculate normal stresses at extreme fibers
    normal_stress_top = calculate_normal_stress(bending_moment, -y_top, moment_of_inertia)  # Compression (top)
    normal_stress_bottom = calculate_normal_stress(bending_moment, -y_bottom, moment_of_inertia)  # Tension (bottom)

    # Calculate maximum shear stress (for rectangular section, it's at the neutral axis)
    # For a rectangular section, Q at neutral axis = width * height² / 8
    first_moment = width * height ** 2 / 8
    max_shear_stress = calculate_shear_stress(shear_force, first_moment, width, moment_of_inertia)

    # Get material yield strength
    yield_strength = MATERIAL_YIELD_STRENGTHS.get(material, 250e6)  # Default to steel if not found

    # Calculate safety factors
    sf_compression = np.ones_like(normal_stress_top)
    sf_tension = np.ones_like(normal_stress_bottom)
    sf_shear = np.ones_like(max_shear_stress)

    # Avoid division by zero
    nonzero_compression = np.abs(normal_stress_top) > 1e-10
    nonzero_tension = np.abs(normal_stress_bottom) > 1e-10
    nonzero_shear = np.abs(max_shear_stress) > 1e-10

    sf_compression[nonzero_compression] = yield_strength / np.abs(normal_stress_top[nonzero_compression])
    sf_tension[nonzero_tension] = yield_strength / np.abs(normal_stress_bottom[nonzero_tension])
    sf_shear[nonzero_shear] = (yield_strength / np.sqrt(3)) / np.abs(
        max_shear_stress[nonzero_shear])  # Von Mises yield criterion

    # Calculate minimum safety factor
    min_sf = np.minimum(np.minimum(sf_compression, sf_tension), sf_shear)

    # Determine critical sections
    critical_x_normal = x[np.argmin(np.minimum(sf_compression, sf_tension))]
    critical_x_shear = x[np.argmin(sf_shear)]

    return {
        "x": x,
        "normal_stress_top": normal_stress_top,
        "normal_stress_bottom": normal_stress_bottom,
        "max_shear_stress": max_shear_stress,
        "safety_factor_compression": sf_compression,
        "safety_factor_tension": sf_tension,
        "safety_factor_shear": sf_shear,
        "min_safety_factor": min_sf,
        "critical_x_normal": critical_x_normal,
        "critical_x_shear": critical_x_shear,
        "yield_strength": yield_strength
    }

# ----------------------------------------------------------------------
# 7. VISUALIZATION
# ----------------------------------------------------------------------
def plot_beam_diagram(beam_type, beam_length, loads):
    """
    Generate a static plot of the beam with loads.

    Parameters:
    -----------
    beam_type : str
        Type of beam (e.g., "Simply Supported", "Cantilever")
    beam_length : float
        Length of the beam in meters
    loads : list
        List of load dictionaries containing load information

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the beam diagram
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot beam
    ax.plot([0, beam_length], [0, 0], 'k-', linewidth=3)

    # Add supports based on beam type
    if beam_type == "Simply Supported":
        # Triangle at left end
        ax.plot([0, 0], [0, -0.5], 'k-', linewidth=2)
        ax.plot([0, -0.2], [-0.5, 0], 'k-', linewidth=2)
        ax.plot([0, 0.2], [-0.5, 0], 'k-', linewidth=2)

        # Circle at right end
        circle = plt.Circle((beam_length, -0.25), 0.25, fill=False, color='k', linewidth=2)
        ax.add_artist(circle)
        ax.plot([beam_length, beam_length], [0, -0.25], 'k-', linewidth=2)

    elif beam_type == "Cantilever":
        # Fixed support at left end
        ax.plot([0, 0], [-1, 1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [-1, -1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [1, 1], 'k-', linewidth=3)
        ax.plot([-0.2, -0.2], [-1, 1], 'k-', linewidth=3)

    elif beam_type == "Fixed-Fixed":
        # Fixed support at both ends
        ax.plot([0, 0], [-1, 1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [-1, -1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [1, 1], 'k-', linewidth=3)
        ax.plot([-0.2, -0.2], [-1, 1], 'k-', linewidth=3)

        ax.plot([beam_length, beam_length], [-1, 1], 'k-', linewidth=3)
        ax.plot([beam_length, beam_length + 0.2], [-1, -1], 'k-', linewidth=3)
        ax.plot([beam_length, beam_length + 0.2], [1, 1], 'k-', linewidth=3)
        ax.plot([beam_length + 0.2, beam_length + 0.2], [-1, 1], 'k-', linewidth=3)

    elif beam_type == "Fixed-Supported":
        # Fixed support at left end
        ax.plot([0, 0], [-1, 1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [-1, -1], 'k-', linewidth=3)
        ax.plot([0, -0.2], [1, 1], 'k-', linewidth=3)
        ax.plot([-0.2, -0.2], [-1, 1], 'k-', linewidth=3)

        # Simple support at right end
        ax.plot([beam_length, beam_length], [0, -0.5], 'k-', linewidth=2)
        ax.plot([beam_length, beam_length - 0.2], [-0.5, 0], 'k-', linewidth=2)
        ax.plot([beam_length, beam_length + 0.2], [-0.5, 0], 'k-', linewidth=2)

    elif beam_type == "Supported-Supported":
        # Simple supports at both ends
        ax.plot([0, 0], [0, -0.5], 'k-', linewidth=2)
        ax.plot([0, -0.2], [-0.5, 0], 'k-', linewidth=2)
        ax.plot([0, 0.2], [-0.5, 0], 'k-', linewidth=2)

        ax.plot([beam_length, beam_length], [0, -0.5], 'k-', linewidth=2)
        ax.plot([beam_length, beam_length - 0.2], [-0.5, 0], 'k-', linewidth=2)
        ax.plot([beam_length, beam_length + 0.2], [-0.5, 0], 'k-', linewidth=2)

    elif beam_type == "Overhanging":
        # Simple supports at 20% and 80% of span (for demonstration)
        support_1 = 0.2 * beam_length
        support_2 = 0.8 * beam_length

        ax.plot([support_1, support_1], [0, -0.5], 'k-', linewidth=2)
        ax.plot([support_1, support_1 - 0.2], [-0.5, 0], 'k-', linewidth=2)
        ax.plot([support_1, support_1 + 0.2], [-0.5, 0], 'k-', linewidth=2)

        circle = plt.Circle((support_2, -0.25), 0.25, fill=False, color='k', linewidth=2)
        ax.add_artist(circle)
        ax.plot([support_2, support_2], [0, -0.25], 'k-', linewidth=2)

    # Add loads
    for load in loads:
        load_type = load["type"]

        if load_type == "Point Load":
            position = load["position"]
            magnitude = load["magnitude"]

            # Determine direction (up or down)
            direction = -1 if magnitude >= 0 else 1

            # Draw arrow
            ax.arrow(position, 0, 0, direction * 1, head_width=0.1, head_length=0.2,
                     fc='r', ec='r', linewidth=2)

            # Add text label
            ax.text(position, direction * 1.2, f"{abs(magnitude)} N",
                    ha='center', va='center', color='r')

        elif load_type == "UDL":
            start = load["start"]
            end = load["end"]
            magnitude = load["magnitude"]

            # Determine direction (up or down)
            direction = -1 if magnitude >= 0 else 1

            # Number of arrows to draw
            num_arrows = min(int((end - start) / (beam_length / 20)) + 1, 10)

            if num_arrows <= 1:
                num_arrows = 2  # Ensure at least 2 arrows

            # Draw arrows
            positions = np.linspace(start, end, num_arrows)
            for pos in positions:
                ax.arrow(pos, 0, 0, direction * 1, head_width=0.1, head_length=0.2,
                         fc='r', ec='r', linewidth=2)

            # Draw line connecting arrow tips
            ax.plot([start, end], [direction * 1, direction * 1], 'r-', linewidth=2)

            # Add text label
            ax.text((start + end) / 2, direction * 1.2, f"{abs(magnitude)} N/m",
                    ha='center', va='center', color='r')

        elif load_type == "Linearly Varying":
            start = load["start"]
            end = load["end"]
            start_magnitude = load["start_magnitude"]
            end_magnitude = load["end_magnitude"]

            # Determine overall direction
            direction = -1 if (start_magnitude + end_magnitude) >= 0 else 1

            # Number of arrows to draw
            num_arrows = min(int((end - start) / (beam_length / 20)) + 1, 10)

            if num_arrows <= 1:
                num_arrows = 2  # Ensure at least 2 arrows

            # Draw arrows with varying lengths
            positions = np.linspace(start, end, num_arrows)
            magnitudes = np.linspace(start_magnitude, end_magnitude, num_arrows)

            # Scale factor for arrow length
            max_magnitude = max(abs(start_magnitude), abs(end_magnitude))
            scale = 1.0 / max_magnitude if max_magnitude > 0 else 1.0

            for pos, mag in zip(positions, magnitudes):
                arrow_length = direction * abs(mag) * scale
                ax.arrow(pos, 0, 0, arrow_length, head_width=0.1, head_length=min(0.2, abs(arrow_length) / 2),
                         fc='r', ec='r', linewidth=2)

            # Draw line connecting arrow tips
            arrow_lengths = [direction * abs(mag) * scale for mag in magnitudes]
            ax.plot(positions, arrow_lengths, 'r-', linewidth=2)

            # Add text labels
            ax.text(start, direction * abs(start_magnitude) * scale * 1.2, f"{abs(start_magnitude)} N/m",
                    ha='center', va='center', color='r')
            ax.text(end, direction * abs(end_magnitude) * scale * 1.2, f"{abs(end_magnitude)} N/m",
                    ha='center', va='center', color='r')

        elif load_type == "Moment":
            position = load["position"]
            magnitude = load["magnitude"]

            # Determine direction of moment (clockwise or counter-clockwise)
            if magnitude >= 0:  # Counter-clockwise positive
                theta = np.linspace(0, 3 * np.pi / 2, 100)
            else:  # Clockwise negative
                theta = np.linspace(0, -3 * np.pi / 2, 100)

            # Size of the moment symbol
            radius = 0.5

            # Draw arc
            x_arc = position + radius * np.cos(theta)
            y_arc = radius * np.sin(theta)
            ax.plot(x_arc, y_arc, 'r-', linewidth=2)

            # Add arrowhead
            if magnitude >= 0:
                ax.arrow(position, radius, -0.1, -0.1, head_width=0.15, head_length=0.15,
                         fc='r', ec='r', linewidth=2)
            else:
                ax.arrow(position, -radius, -0.1, 0.1, head_width=0.15, head_length=0.15,
                         fc='r', ec='r', linewidth=2)

            # Add text label
            y_offset = 1.2 * radius
            ax.text(position, y_offset, f"{abs(magnitude)} N·m",
                    ha='center', va='center', color='r')

    # Set axis properties
    ax.set_xlim(-0.5, beam_length + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Position (m)')
    ax.set_title(f'{beam_type} Beam Diagram')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Hide y-axis ticks and labels
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_shear_force(x, shear_force):
    """
    Generate a static plot of the shear force diagram.

    Parameters:
    -----------
    x : numpy.ndarray
        Array of x positions along the beam
    shear_force : numpy.ndarray
        Array of shear force values corresponding to x positions

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the shear force diagram
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot shear force
    ax.plot(x, shear_force, 'b-', linewidth=2)

    # Fill area
    ax.fill_between(x, shear_force, alpha=0.3, color='b')

    # Set axis properties
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Shear Force (N)')
    ax.set_title('Shear Force Diagram')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add maximum and minimum values
    max_shear = max(shear_force)
    min_shear = min(shear_force)
    abs_max = max(abs(max_shear), abs(min_shear))

    max_idx = np.argmax(shear_force)
    min_idx = np.argmin(shear_force)

    if max_shear > 0:
        ax.annotate(f'Max: {max_shear:.2f} N',
                    xy=(x[max_idx], max_shear),
                    xytext=(x[max_idx], max_shear + 0.1 * abs_max),
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->'))

    if min_shear < 0:
        ax.annotate(f'Min: {min_shear:.2f} N',
                    xy=(x[min_idx], min_shear),
                    xytext=(x[min_idx], min_shear - 0.1 * abs_max),
                    ha='center', va='top',
                    arrowprops=dict(arrowstyle='->'))

    fig.tight_layout()
    return fig


def plot_bending_moment(x, bending_moment):
    """
    Generate a static plot of the bending moment diagram.

    Parameters:
    -----------
    x : numpy.ndarray
        Array of x positions along the beam
    bending_moment : numpy.ndarray
        Array of bending moment values corresponding to x positions

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the bending moment diagram
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot bending moment
    ax.plot(x, bending_moment, 'g-', linewidth=2)

    # Fill area
    ax.fill_between(x, bending_moment, alpha=0.3, color='g')

    # Set axis properties
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Bending Moment (N·m)')
    ax.set_title('Bending Moment Diagram')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add maximum and minimum values
    max_moment = max(bending_moment)
    min_moment = min(bending_moment)
    abs_max = max(abs(max_moment), abs(min_moment))

    max_idx = np.argmax(bending_moment)
    min_idx = np.argmin(bending_moment)

    if max_moment > 0:
        ax.annotate(f'Max: {max_moment:.2f} N·m',
                    xy=(x[max_idx], max_moment),
                    xytext=(x[max_idx], max_moment + 0.1 * abs_max),
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->'))

    if min_moment < 0:
        ax.annotate(f'Min: {min_moment:.2f} N·m',
                    xy=(x[min_idx], min_moment),
                    xytext=(x[min_idx], min_moment - 0.1 * abs_max),
                    ha='center', va='top',
                    arrowprops=dict(arrowstyle='->'))

    fig.tight_layout()
    return fig


def plot_deflection(x, deflection):
    """
    Generate a static plot of the deflection diagram.

    Parameters:
    -----------
    x : numpy.ndarray
        Array of x positions along the beam
    deflection : numpy.ndarray
        Array of deflection values corresponding to x positions

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the deflection diagram
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Convert to mm for better visualization
    deflection_mm = deflection * 1000

    # Plot deflection
    ax.plot(x, deflection_mm, 'r-', linewidth=2)

    # Fill area
    ax.fill_between(x, deflection_mm, alpha=0.3, color='r')

    # Set axis properties
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Deflection (mm)')
    ax.set_title('Deflection Diagram')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add maximum and minimum values
    max_defl = max(deflection_mm)
    min_defl = min(deflection_mm)
    abs_max = max(abs(max_defl), abs(min_defl))

    max_idx = np.argmax(deflection_mm)
    min_idx = np.argmin(deflection_mm)

    if max_defl > 0:
        ax.annotate(f'Max: {max_defl:.2f} mm',
                    xy=(x[max_idx], max_defl),
                    xytext=(x[max_idx], max_defl + 0.1 * abs_max),
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->'))

    if min_defl < 0:
        ax.annotate(f'Min: {min_defl:.2f} mm',
                    xy=(x[min_idx], min_defl),
                    xytext=(x[min_idx], min_defl - 0.1 * abs_max),
                    ha='center', va='top',
                    arrowprops=dict(arrowstyle='->'))

    fig.tight_layout()
    return fig


def create_interactive_plots(x, shear_force, bending_moment, deflection):
    """
    Create interactive plots using Plotly for beam analysis results.

    Parameters:
    -----------
    x : numpy.ndarray
        Array of x positions along the beam
    shear_force : numpy.ndarray
        Array of shear force values
    bending_moment : numpy.ndarray
        Array of bending moment values
    deflection : numpy.ndarray
        Array of deflection values

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing interactive plots
    """
    # Convert deflection to mm for better visualization
    deflection_mm = deflection * 1000  # Convert m to mm

    # Create figure with subplots
    fig = go.Figure()

    # Add traces for each result
    fig.add_trace(go.Scatter(
        x=x, y=shear_force,
        mode='lines',
        name='Shear Force (N)',
        line=dict(color='blue', width=2),
        hovertemplate='Position: %{x:.2f} m<br>Shear Force: %{y:.2f} N'
    ))

    fig.add_trace(go.Scatter(
        x=x, y=bending_moment,
        mode='lines',
        name='Bending Moment (N·m)',
        line=dict(color='green', width=2),
        visible='legendonly',  # Hidden by default
        hovertemplate='Position: %{x:.2f} m<br>Bending Moment: %{y:.2f} N·m'
    ))

    fig.add_trace(go.Scatter(
        x=x, y=deflection_mm,
        mode='lines',
        name='Deflection (mm)',
        line=dict(color='red', width=2),
        visible='legendonly',  # Hidden by default
        hovertemplate='Position: %{x:.2f} m<br>Deflection: %{y:.2f} mm'
    ))

    # Update layout
    fig.update_layout(
        title="Beam Analysis Results",
        xaxis_title="Position (m)",
        hovermode="closest",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"visible": [True, False, False]},
                              {"yaxis.title.text": "Shear Force (N)"}],
                        label="Shear Force",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False]},
                              {"yaxis.title.text": "Bending Moment (N·m)"}],
                        label="Bending Moment",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False, True]},
                              {"yaxis.title.text": "Deflection (mm)"}],
                        label="Deflection",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True, True]},
                              {"yaxis.title.text": "Value"}],
                        label="Show All",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    return fig

# ----------------------------------------------------------------------
# 8. REPORT GENERATOR
# ----------------------------------------------------------------------
def generate_report(project_name, engineer_name, beam_type, beam_length, loads, results,
                    include_diagrams=True, include_equations=True, include_max_values=True,
                    include_reactions=True, elastic_modulus=200e9, moment_of_inertia=2e-5):
    """
    Generate a PDF report for the beam analysis.

    Parameters:
    -----------
    project_name : str
        Name of the project
    engineer_name : str
        Name of the engineer
    beam_type : str
        Type of beam
    beam_length : float
        Length of the beam in meters
    loads : list
        List of load dictionaries
    results : dict
        Results from the beam analysis calculation
    include_diagrams : bool
        Whether to include diagrams in the report
    include_equations : bool
        Whether to include equations in the report
    include_max_values : bool
        Whether to include maximum values in the report
    include_reactions : bool
        Whether to include support reactions in the report
    elastic_modulus : float
        Elastic modulus of the beam material
    moment_of_inertia : float
        Moment of inertia of the beam cross-section

    Returns:
    --------
    bytes
        PDF report as bytes
    """
    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

    # Get sample styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        leading=24,
        textColor=colors.darkblue,
        alignment=1
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=14,
        leading=18,
        textColor=colors.darkblue
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.darkblue
    )

    normal_style = styles['Normal']

    # Initialize story (elements to add to the PDF)
    story = []

    # Add title
    story.append(Paragraph(project_name, title_style))
    story.append(Spacer(1, 0.25 * inch))

    # Add report details
    story.append(Paragraph("Beam Analysis Report", heading1_style))
    story.append(Spacer(1, 0.1 * inch))

    # Add date and engineer
    current_date = datetime.now().strftime("%Y-%m-%d")
    detail_data = [
        ["Date:", current_date],
        ["Engineer:", engineer_name if engineer_name else ""]
    ]

    detail_table = Table(detail_data, colWidths=[1.5 * inch, 4 * inch])
    detail_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))

    story.append(detail_table)
    story.append(Spacer(1, 0.25 * inch))

    # Add beam parameters
    story.append(Paragraph("Beam Parameters", heading1_style))
    story.append(Spacer(1, 0.1 * inch))

    # Create beam parameters table
    beam_params = [
        ["Beam Type:", beam_type],
        ["Beam Length:", f"{beam_length} m"],
        ["Elastic Modulus (E):", f"{elastic_modulus:.2e} N/m²"],
        ["Moment of Inertia (I):", f"{moment_of_inertia:.2e} m⁴"]
    ]

    param_table = Table(beam_params, colWidths=[2.5 * inch, 3 * inch])
    param_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))

    story.append(param_table)
    story.append(Spacer(1, 0.25 * inch))

    # Add loads section
    story.append(Paragraph("Applied Loads", heading1_style))
    story.append(Spacer(1, 0.1 * inch))

    if not loads:
        story.append(Paragraph("No loads applied.", normal_style))
    else:
        load_data = []

        # Create header
        load_data.append(["Load Type", "Parameters", "Values"])

        # Add each load
        for i, load in enumerate(loads):
            load_type = load["type"]

            if load_type == "Point Load":
                params = "Position<br/>Magnitude"
                values = f"{load['position']} m<br/>{load['magnitude']} N"

            elif load_type == "UDL":
                params = "Start Position<br/>End Position<br/>Magnitude"
                values = f"{load['start']} m<br/>{load['end']} m<br/>{load['magnitude']} N/m"

            elif load_type == "Linearly Varying":
                params = "Start Position<br/>End Position<br/>Start Magnitude<br/>End Magnitude"
                values = f"{load['start']} m<br/>{load['end']} m<br/>{load['start_magnitude']} N/m<br/>{load['end_magnitude']} N/m"

            elif load_type == "Moment":
                params = "Position<br/>Magnitude"
                values = f"{load['position']} m<br/>{load['magnitude']} N·m"

            else:
                params = "Unknown"
                values = "Unknown"

            load_data.append([load_type, Paragraph(params, normal_style), Paragraph(values, normal_style)])

        # Create load table
        load_table = Table(load_data, colWidths=[1.5 * inch, 2 * inch, 2 * inch])
        load_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        story.append(load_table)

    story.append(Spacer(1, 0.25 * inch))

    # Add support reactions if requested
    if include_reactions and results is not None:
        story.append(Paragraph("Support Reactions", heading1_style))
        story.append(Spacer(1, 0.1 * inch))

        # Create reactions table
        reaction_data = [["Support", "Value"]]

        for support, value in results["reactions"].items():
            # Format the support name to look better
            if support == "R_A":
                support_label = "Reaction A"
            elif support == "R_B":
                support_label = "Reaction B"
            elif support == "M_A":
                support_label = "Moment A"
            elif support == "M_B":
                support_label = "Moment B"
            else:
                support_label = support

            reaction_data.append([support_label, f"{value:.2f} {'N' if 'R_' in support else 'N·m'}"])

        reaction_table = Table(reaction_data, colWidths=[2.5 * inch, 3 * inch])
        reaction_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))

        story.append(reaction_table)
        story.append(Spacer(1, 0.25 * inch))

    # Add maximum values if requested
    if include_max_values and results is not None:
        story.append(Paragraph("Maximum Values", heading1_style))
        story.append(Spacer(1, 0.1 * inch))

        # Calculate maximum values
        max_shear = max(abs(min(results["shear_force"])), abs(max(results["shear_force"])))
        max_moment = max(abs(min(results["bending_moment"])), abs(max(results["bending_moment"])))
        max_deflection = max(abs(min(results["deflection"])), abs(max(results["deflection"])))

        # Create max values table
        max_data = [
            ["Maximum Shear Force:", f"{max_shear:.2f} N"],
            ["Maximum Bending Moment:", f"{max_moment:.2f} N·m"],
            ["Maximum Deflection:", f"{max_deflection * 1000:.2f} mm"]
        ]

        max_table = Table(max_data, colWidths=[2.5 * inch, 3 * inch])
        max_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))

        story.append(max_table)
        story.append(Spacer(1, 0.25 * inch))

    # Add diagrams if requested
    if include_diagrams and results is not None:
        story.append(PageBreak())
        story.append(Paragraph("Analysis Diagrams", heading1_style))
        story.append(Spacer(1, 0.1 * inch))

        # Function to convert matplotlib figure to ReportLab image
        def fig_to_img(fig, width=6 * inch):
            buf = io.BytesIO()
            # Set a fixed figure size before saving
            fig.set_size_inches(8, 4)  # Set a reasonable aspect ratio
            # Increase DPI for better quality while keeping file size reasonable
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            # Calculate height while maintaining aspect ratio
            img = RLImage(buf, width=width, height=width * 0.5)  # Set height to maintain 2:1 aspect ratio
            return img

        # Add beam diagram
        story.append(Paragraph("Beam Diagram", heading2_style))
        story.append(Spacer(1, 0.1 * inch))

        beam_fig = plot_beam_diagram(beam_type, beam_length, loads)
        story.append(fig_to_img(beam_fig))
        story.append(Spacer(1, 0.2 * inch))

        # Add shear force diagram
        story.append(Paragraph("Shear Force Diagram", heading2_style))
        story.append(Spacer(1, 0.1 * inch))

        shear_fig = plot_shear_force(results["x"], results["shear_force"])
        story.append(fig_to_img(shear_fig))
        story.append(Spacer(1, 0.2 * inch))

        # Add bending moment diagram
        story.append(Paragraph("Bending Moment Diagram", heading2_style))
        story.append(Spacer(1, 0.1 * inch))

        moment_fig = plot_bending_moment(results["x"], results["bending_moment"])
        story.append(fig_to_img(moment_fig))
        story.append(Spacer(1, 0.2 * inch))

        # Add deflection diagram
        story.append(Paragraph("Deflection Diagram", heading2_style))
        story.append(Spacer(1, 0.1 * inch))

        deflection_fig = plot_deflection(results["x"], results["deflection"])
        story.append(fig_to_img(deflection_fig))

    # Add disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Disclaimer and Notes", heading1_style))
    story.append(Spacer(1, 0.1 * inch))

    disclaimer_text = """
    This report is generated automatically based on the provided inputs and should be 
    reviewed by a qualified structural engineer before being used for design purposes. 
    The analysis is based on simplified beam theory assumptions, including:

    1. Linear elastic material behavior
    2. Small deflections
    3. Constant cross-section throughout the beam
    4. No consideration of material failure criteria

    The user is responsible for verifying the results and ensuring all appropriate 
    safety factors are applied for the specific application.
    """

    story.append(Paragraph(disclaimer_text, normal_style))

    # Build the PDF
    doc.build(story)

    # Get the PDF content
    pdf_content = buffer.getvalue()
    buffer.close()

    return pdf_content

# ----------------------------------------------------------------------
# 9. MAIN APPLICATION
# ----------------------------------------------------------------------
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Structural Beam Analysis",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App title and description
    st.title("Structural Beam Analysis")
    st.markdown("""
    This application helps engineers analyze structural beams under various loading conditions.
    Input your beam parameters, apply loads, and get instant results including reactions, 
    shear force, bending moment, and deflection diagrams.
    """)

    # Initialize session state variables if they don't exist
    if 'beam_length' not in st.session_state:
        st.session_state.beam_length = 5.0
    if 'beam_type' not in st.session_state:
        st.session_state.beam_type = "Simply Supported"
    if 'loads' not in st.session_state:
        st.session_state.loads = []
    if 'elastic_modulus' not in st.session_state:
        st.session_state.elastic_modulus = 200e9  # Default: Steel (N/m²)
    if 'moment_of_inertia' not in st.session_state:
        st.session_state.moment_of_inertia = 2e-5  # Default: m⁴
    if 'current_load_type' not in st.session_state:
        st.session_state.current_load_type = "Point Load"
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'saved_configurations' not in st.session_state:
        st.session_state.saved_configurations = {}
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    if 'comparison_active' not in st.session_state:
        st.session_state.comparison_active = False
    if 'load_types' not in st.session_state:
        st.session_state.load_types = []  # List to track the type (Dead, Live, etc.) of each load

    # Sidebar for beam configuration
    with st.sidebar:
        st.header("Beam Configuration")

        # Beam type selection with icons
        beam_options = list(BEAM_ICONS.keys())
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_beam = st.selectbox("Beam Type", beam_options,
                                         index=beam_options.index(st.session_state.beam_type))
        with col2:
            st.markdown(BEAM_ICONS[selected_beam], unsafe_allow_html=True)

        st.session_state.beam_type = selected_beam

        # Beam dimensions
        st.session_state.beam_length = st.number_input("Beam Length (m)", min_value=0.1, max_value=100.0,
                                                       value=st.session_state.beam_length, step=0.1)

        # Material and cross-section properties
        st.subheader("Material & Cross-Section")

        # Get all section categories and names
        section_names_by_category = get_section_names_by_category()

        # Cross-section type selection
        use_standard_section = st.checkbox("Use Standard Cross-Section", value=False)

        if use_standard_section:
            # Standard section selection
            section_category = st.selectbox(
                "Section Type",
                list(section_names_by_category.keys()),
                index=0
            )

            section_name = st.selectbox(
                "Section Size",
                section_names_by_category[section_category],
                index=0
            )

            # Get properties of selected section
            section_properties = get_section_properties(section_name)

            # Display section properties
            if section_properties:
                st.write("Section Properties:")

                # Create two columns for properties display
                prop_col1, prop_col2 = st.columns(2)

                with prop_col1:
                    if "height" in section_properties:
                        st.write(f"Height: {section_properties['height']} m")
                    if "width" in section_properties:
                        st.write(f"Width: {section_properties['width']} m")
                    if "diameter" in section_properties:
                        st.write(f"Diameter: {section_properties['diameter']} m")

                with prop_col2:
                    if "area" in section_properties:
                        st.write(f"Area: {section_properties['area']} m²")
                    if "material" in section_properties:
                        st.write(f"Material: {section_properties['material']}")

                # Set elastic modulus based on material
                material = section_properties.get("material", "Steel")
                if material == "Steel":
                    st.session_state.elastic_modulus = 200e9
                elif material == "Concrete":
                    st.session_state.elastic_modulus = 30e9
                elif material == "Aluminum":
                    st.session_state.elastic_modulus = 70e9
                elif material == "Timber":
                    st.session_state.elastic_modulus = 11e9

                # Get moment of inertia
                if "diameter" in section_properties or "Circular" in section_name:
                    moment_of_inertia = section_properties["I"]
                else:
                    moment_of_inertia = section_properties["I_xx"]  # Use strong axis by default

                st.session_state.moment_of_inertia = moment_of_inertia
                st.write(f"Moment of Inertia: {moment_of_inertia:.2e} m⁴")
        else:
            # Manual material selection
            material = st.selectbox(
                "Material Type",
                ["Steel", "Concrete", "Aluminum", "Timber", "Custom"],
                index=0
            )

            # Set material properties based on selection
            if material == "Steel":
                st.session_state.elastic_modulus = 200e9
            elif material == "Concrete":
                st.session_state.elastic_modulus = 30e9
            elif material == "Aluminum":
                st.session_state.elastic_modulus = 70e9
            elif material == "Timber":
                st.session_state.elastic_modulus = 11e9

            if material == "Custom":
                st.session_state.elastic_modulus = st.number_input("Elastic Modulus (N/m²)", min_value=1e6,
                                                                   max_value=1e12,
                                                                   value=float(st.session_state.elastic_modulus),
                                                                   format="%.2e")

            st.session_state.moment_of_inertia = st.number_input("Moment of Inertia (m⁴)", min_value=1e-8,
                                                                 max_value=1.0,
                                                                 value=float(st.session_state.moment_of_inertia),
                                                                 format="%.2e")

        # Save and load configurations
        st.subheader("Save/Load Configuration")
        config_name = st.text_input("Configuration Name")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save"):
                if config_name:
                    st.session_state.saved_configurations[config_name] = {
                        "beam_type": st.session_state.beam_type,
                        "beam_length": st.session_state.beam_length,
                        "elastic_modulus": st.session_state.elastic_modulus,
                        "moment_of_inertia": st.session_state.moment_of_inertia,
                        "loads": st.session_state.loads
                    }
                    st.success(f"Configuration '{config_name}' saved!")
                else:
                    st.warning("Please enter a configuration name")

        with col2:
            if st.session_state.saved_configurations:
                selected_config = st.selectbox("Select Configuration",
                                               list(st.session_state.saved_configurations.keys()))
                if st.button("Load"):
                    config = st.session_state.saved_configurations[selected_config]
                    st.session_state.beam_type = config["beam_type"]
                    st.session_state.beam_length = config["beam_length"]
                    st.session_state.elastic_modulus = config["elastic_modulus"]
                    st.session_state.moment_of_inertia = config["moment_of_inertia"]
                    st.session_state.loads = config["loads"]
                    st.success(f"Configuration '{selected_config}' loaded!")
                    st.session_state.show_results = False
                    st.rerun()

    # Main content area
    # Tab layout for different sections
    tab1, tab2, tab3 = st.tabs(["Load Definition", "Results", "Report"])

    # Tab 1: Load Definition
    with tab1:
        st.header("Load Definition")

        # Load type selection with icons
        load_options = list(LOAD_ICONS.keys())
        col1, col2 = st.columns([3, 1])
        with col1:
            load_type = st.selectbox("Load Type", load_options,
                                     index=load_options.index(st.session_state.current_load_type))
        with col2:
            st.markdown(LOAD_ICONS[load_type], unsafe_allow_html=True)

        st.session_state.current_load_type = load_type

        # Load parameters based on type
        with st.form("load_form"):
            # Load classification for design codes
            load_classification = st.selectbox(
                "Load Classification",
                LOAD_TYPES,
                index=0,
                help="Classify this load for use with design code load combinations"
            )

            if load_type == "Point Load":
                magnitude = st.number_input("Magnitude (N)", min_value=-1000000.0, max_value=1000000.0, value=1000.0,
                                            step=100.0)
                position = st.slider("Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                     value=st.session_state.beam_length / 2, step=0.1)
                load_data = {"type": "Point Load", "magnitude": magnitude, "position": position}

            elif load_type == "Uniform Distributed Load (UDL)":
                magnitude = st.number_input("Magnitude (N/m)", min_value=-1000000.0, max_value=1000000.0, value=1000.0,
                                            step=100.0)
                start = st.slider("Start Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                  value=0.0, step=0.1)
                end = st.slider("End Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                value=float(st.session_state.beam_length), step=0.1)

                # Input validation
                valid_input = True
                if start >= end:
                    st.error("Start position must be less than end position.")
                    valid_input = False

                load_data = {"type": "UDL", "magnitude": magnitude, "start": start, "end": end, "valid": valid_input}

            elif load_type == "Linearly Varying Load":
                start_magnitude = st.number_input("Start Magnitude (N/m)", min_value=-1000000.0, max_value=1000000.0,
                                                  value=0.0, step=100.0)
                end_magnitude = st.number_input("End Magnitude (N/m)", min_value=-1000000.0, max_value=1000000.0,
                                                value=1000.0, step=100.0)
                start = st.slider("Start Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                  value=0.0, step=0.1)
                end = st.slider("End Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                value=float(st.session_state.beam_length), step=0.1)

                # Input validation
                valid_input = True
                if start >= end:
                    st.error("Start position must be less than end position.")
                    valid_input = False

                load_data = {"type": "Linearly Varying", "start_magnitude": start_magnitude,
                             "end_magnitude": end_magnitude, "start": start, "end": end, "valid": valid_input}

            elif load_type == "Moment":
                magnitude = st.number_input("Magnitude (N·m)", min_value=-1000000.0, max_value=1000000.0, value=1000.0,
                                            step=100.0)
                position = st.slider("Position (m)", min_value=0.0, max_value=float(st.session_state.beam_length),
                                     value=st.session_state.beam_length / 2, step=0.1)
                load_data = {"type": "Moment", "magnitude": magnitude, "position": position}

            submit_load = st.form_submit_button("Add Load")

        if submit_load:
            # Check if the load data is valid before adding
            if "valid" in load_data and not load_data["valid"]:
                st.error("Cannot add load with invalid parameters. Please correct the errors.")
            else:
                # Remove 'valid' key if it exists before adding to loads
                if "valid" in load_data:
                    del load_data["valid"]

                st.session_state.loads.append(load_data)
                st.session_state.load_types.append(load_classification)  # Store the load classification
                st.success(f"{load_type} added successfully! (Classified as {load_classification})")
                st.session_state.show_results = False

        # Display current loads
        st.subheader("Current Loads")
        if not st.session_state.loads:
            st.info("No loads added yet. Add loads using the form above.")
        else:
            for i, load in enumerate(st.session_state.loads):
                load_type = load["type"]
                if load_type == "Point Load":
                    st.write(f"{i + 1}. Point Load: {load['magnitude']} N at position {load['position']} m")
                elif load_type == "UDL":
                    st.write(f"{i + 1}. UDL: {load['magnitude']} N/m from {load['start']} m to {load['end']} m")
                elif load_type == "Linearly Varying":
                    st.write(
                        f"{i + 1}. Linearly Varying Load: {load['start_magnitude']} N/m to {load['end_magnitude']} N/m from {load['start']} m to {load['end']} m")
                elif load_type == "Moment":
                    st.write(f"{i + 1}. Moment: {load['magnitude']} N·m at position {load['position']} m")

                if st.button(f"Remove Load {i + 1}"):
                    st.session_state.loads.pop(i)
                    # Also remove the corresponding load type if it exists
                    if i < len(st.session_state.load_types):
                        st.session_state.load_types.pop(i)
                    st.session_state.show_results = False
                    st.rerun()

        # Clear all loads
        if st.session_state.loads and st.button("Clear All Loads"):
            st.session_state.loads = []
            st.session_state.load_types = []
            st.session_state.show_results = False
            st.rerun()

        # Load combinations section
        if st.session_state.loads:
            st.subheader("Load Combinations")

            # Design code selection
            design_codes = get_design_codes()
            selected_code = st.selectbox("Design Code", design_codes)

            # Load combinations for the selected code
            combinations = get_combinations_for_code(selected_code)
            if combinations:
                st.write(f"Available load combinations for {selected_code}:")

                # Display combinations and allow selection
                cols = st.columns(2)
                with cols[0]:
                    selected_combination = st.selectbox(
                        "Select a load combination",
                        list(combinations.keys())
                    )

                # Show description and factors
                with cols[1]:
                    if selected_combination:
                        st.write(f"Description: {combinations[selected_combination]['description']}")
                        st.write("Factors:")
                        for load_type, factor in combinations[selected_combination]['factors'].items():
                            st.write(f"  • {load_type}: {factor}")

                # Apply load combination
                if st.button(f"Apply {selected_combination} Combination"):
                    # Make sure we have load types assigned to each load
                    if len(st.session_state.load_types) != len(st.session_state.loads):
                        # Handle case where load types might be missing (backward compatibility)
                        if len(st.session_state.load_types) == 0:
                            # Assign all loads as "Dead Load" if no types exist
                            st.session_state.load_types = ["Dead Load"] * len(st.session_state.loads)
                        else:
                            # Extend load_types list with "Dead Load" as default
                            st.session_state.load_types.extend(
                                ["Dead Load"] * (len(st.session_state.loads) - len(st.session_state.load_types)))

                    # Get the factors for the selected combination
                    combination_factors = combinations[selected_combination]['factors']

                    # Apply the combination
                    combined_loads = apply_load_combination(
                        st.session_state.loads,
                        st.session_state.load_types,
                        combination_factors
                    )

                    if combined_loads:
                        # Calculate with the combined loads
                        try:
                            results = calculate_beam_response(
                                st.session_state.beam_type,
                                st.session_state.beam_length,
                                combined_loads,
                                st.session_state.elastic_modulus,
                                st.session_state.moment_of_inertia
                            )
                            st.session_state.results = results
                            st.session_state.show_results = True
                            st.success(
                                f"Load combination '{selected_combination}' applied and calculated. View results in the Results tab.")
                        except Exception as e:
                            st.error(f"Error during calculation with load combination: {str(e)}")
                    else:
                        st.warning(
                            "No loads were included in this combination. Check that your load classifications match the factors in the combination.")
            else:
                st.write("No combinations available for the selected design code.")

        # Calculate results
        if st.button("Calculate Results (Without Combinations)"):
            if not st.session_state.loads:
                st.warning("Please add at least one load before calculating results.")
            else:
                # Calculate beam response
                try:
                    results = calculate_beam_response(
                        st.session_state.beam_type,
                        st.session_state.beam_length,
                        st.session_state.loads,
                        st.session_state.elastic_modulus,
                        st.session_state.moment_of_inertia
                    )
                    st.session_state.results = results
                    st.session_state.show_results = True
                    st.success("Calculation completed. View results in the Results tab.")
                except Exception as e:
                    st.error(f"Error during calculation: {str(e)}")

    # Tab 2: Results
    with tab2:
        if st.session_state.show_results and st.session_state.results is not None:
            # Add option to save current result for comparison
            st.header("Analysis Results")

            results_col1, results_col2 = st.columns([2, 1])
            with results_col1:
                if st.button("Save Current Result for Comparison"):
                    # Generate a unique name for this configuration
                    config_name = f"{st.session_state.beam_type} - {len(st.session_state.loads)} loads"

                    # If this name already exists, add a counter
                    base_name = config_name
                    counter = 1
                    while config_name in st.session_state.comparison_results:
                        config_name = f"{base_name} ({counter})"
                        counter += 1

                    # Save the current results and configuration
                    st.session_state.comparison_results[config_name] = {
                        "beam_type": st.session_state.beam_type,
                        "beam_length": st.session_state.beam_length,
                        "elastic_modulus": st.session_state.elastic_modulus,
                        "moment_of_inertia": st.session_state.moment_of_inertia,
                        "loads": st.session_state.loads.copy(),
                        "results": st.session_state.results.copy()
                    }
                    st.success(f"Result saved as '{config_name}' for comparison")
                    st.session_state.comparison_active = True
                    st.rerun()

            with results_col2:
                if st.session_state.comparison_results and st.button("Clear All Comparisons"):
                    st.session_state.comparison_results = {}
                    st.session_state.comparison_active = False
                    st.success("All comparisons cleared")
                    st.rerun()

            # Display comparisons if there are saved results
            if st.session_state.comparison_active and len(st.session_state.comparison_results) > 0:
                st.subheader("Result Comparisons")

                # Create multiselect for configurations to compare
                configs_to_compare = st.multiselect(
                    "Select configurations to compare",
                    options=list(st.session_state.comparison_results.keys()),
                    default=[list(st.session_state.comparison_results.keys())[0]]
                )

                if configs_to_compare:
                    # Add current result to the comparison
                    include_current = st.checkbox("Include current result in comparison", value=True)

                    # Create comparison tabs
                    comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Shear Force", "Bending Moment", "Deflection"])

                    # Setup comparison data
                    comparison_data = {}

                    # Add selected saved configurations
                    for config_name in configs_to_compare:
                        config_data = st.session_state.comparison_results[config_name]
                        comparison_data[config_name] = {
                            "x": config_data["results"]["x"],
                            "shear_force": config_data["results"]["shear_force"],
                            "bending_moment": config_data["results"]["bending_moment"],
                            "deflection": config_data["results"]["deflection"]
                        }

                    # Add current configuration if selected
                    if include_current:
                        comparison_data["Current"] = {
                            "x": st.session_state.results["x"],
                            "shear_force": st.session_state.results["shear_force"],
                            "bending_moment": st.session_state.results["bending_moment"],
                            "deflection": st.session_state.results["deflection"]
                        }

                    # Shear Force Comparison
                    with comp_tab1:
                        # Create a plotly figure for comparison
                        fig = go.Figure()

                        for name, data in comparison_data.items():
                            fig.add_trace(go.Scatter(
                                x=data["x"],
                                y=data["shear_force"],
                                mode='lines',
                                name=name
                            ))

                        fig.update_layout(
                            title="Shear Force Comparison",
                            xaxis_title="Position (m)",
                            yaxis_title="Shear Force (N)",
                            legend_title="Configurations",
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Table with maximum values
                        st.subheader("Maximum Shear Force Comparison")
                        max_shear_data = {name: max(abs(min(data["shear_force"])), abs(max(data["shear_force"])))
                                          for name, data in comparison_data.items()}
                        max_shear_df = pd.DataFrame(max_shear_data.items(),
                                                    columns=["Configuration", "Max Shear Force (N)"])
                        max_shear_df = max_shear_df.sort_values(by="Max Shear Force (N)", ascending=False)
                        st.table(max_shear_df)

                    # Bending Moment Comparison
                    with comp_tab2:
                        # Create a plotly figure for comparison
                        fig = go.Figure()

                        for name, data in comparison_data.items():
                            fig.add_trace(go.Scatter(
                                x=data["x"],
                                y=data["bending_moment"],
                                mode='lines',
                                name=name
                            ))

                        fig.update_layout(
                            title="Bending Moment Comparison",
                            xaxis_title="Position (m)",
                            yaxis_title="Bending Moment (N·m)",
                            legend_title="Configurations",
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Table with maximum values
                        st.subheader("Maximum Bending Moment Comparison")
                        max_moment_data = {name: max(abs(min(data["bending_moment"])), abs(max(data["bending_moment"])))
                                           for name, data in comparison_data.items()}
                        max_moment_df = pd.DataFrame(max_moment_data.items(),
                                                     columns=["Configuration", "Max Bending Moment (N·m)"])
                        max_moment_df = max_moment_df.sort_values(by="Max Bending Moment (N·m)", ascending=False)
                        st.table(max_moment_df)

                    # Deflection Comparison
                    with comp_tab3:
                        # Create a plotly figure for comparison
                        fig = go.Figure()

                        for name, data in comparison_data.items():
                            fig.add_trace(go.Scatter(
                                x=data["x"],
                                y=data["deflection"] * 1000,  # Convert to mm for better visualization
                                mode='lines',
                                name=name
                            ))

                        fig.update_layout(
                            title="Deflection Comparison",
                            xaxis_title="Position (m)",
                            yaxis_title="Deflection (mm)",
                            legend_title="Configurations",
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Table with maximum values
                        st.subheader("Maximum Deflection Comparison")
                        max_deflection_data = {
                            name: max(abs(min(data["deflection"])), abs(max(data["deflection"]))) * 1000
                            for name, data in comparison_data.items()}
                        max_deflection_df = pd.DataFrame(max_deflection_data.items(),
                                                         columns=["Configuration", "Max Deflection (mm)"])
                        max_deflection_df = max_deflection_df.sort_values(by="Max Deflection (mm)", ascending=False)
                        st.table(max_deflection_df)

            # Regular results display
            st.subheader("Current Results")

            # Display reactions
            st.subheader("Support Reactions")
            reactions = st.session_state.results["reactions"]
            reaction_df = pd.DataFrame(reactions.items(), columns=["Support", "Value"])
            st.table(reaction_df)

            # Create tabs for different visualization types
            vis_tab1, vis_tab2, vis_tab3 = st.tabs(["Static Plots", "Interactive Plots", "Stress Analysis"])

            with vis_tab1:
                # Static plots using Matplotlib
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Beam Diagram")
                    fig_beam = plot_beam_diagram(
                        st.session_state.beam_type,
                        st.session_state.beam_length,
                        st.session_state.loads
                    )
                    st.pyplot(fig_beam)

                    st.subheader("Shear Force Diagram")
                    fig_shear = plot_shear_force(
                        st.session_state.results["x"],
                        st.session_state.results["shear_force"]
                    )
                    st.pyplot(fig_shear)

                with col2:
                    st.subheader("Bending Moment Diagram")
                    fig_moment = plot_bending_moment(
                        st.session_state.results["x"],
                        st.session_state.results["bending_moment"]
                    )
                    st.pyplot(fig_moment)

                    st.subheader("Deflection Diagram")
                    fig_deflection = plot_deflection(
                        st.session_state.results["x"],
                        st.session_state.results["deflection"]
                    )
                    st.pyplot(fig_deflection)

            with vis_tab2:
                # Interactive plots using Plotly
                st.subheader("Interactive Diagrams")
                fig_interactive = create_interactive_plots(
                    st.session_state.results["x"],
                    st.session_state.results["shear_force"],
                    st.session_state.results["bending_moment"],
                    st.session_state.results["deflection"]
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

            # Stress Analysis Tab
            with vis_tab3:
                st.subheader("Material Stress Analysis")

                # Get cross-section properties
                if "use_standard_section" in locals() and use_standard_section and "section_properties" in locals():
                    current_section_properties = section_properties
                    material_name = section_properties.get("material", "Steel")
                else:
                    # Create default properties for manual cross-section
                    height = st.number_input("Section Height (m)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
                    width = st.number_input("Section Width (m)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

                    # Calculate moment of inertia for rectangular section
                    I_xx = width * height ** 3 / 12

                    current_section_properties = {
                        "height": height,
                        "width": width,
                        "I_xx": I_xx
                    }

                    material_options = ["Steel", "Concrete", "Aluminum", "Timber"]
                    material_name = st.selectbox("Material", material_options, index=0)

                # Calculate stress distribution
                stress_results = calculate_stress_distribution(
                    st.session_state.results,
                    current_section_properties,
                    material_name
                )

                # Display stress results
                st.subheader("Stress Distribution")
                stress_tabs = st.tabs(["Normal Stress", "Shear Stress", "Safety Factors"])

                # Normal Stress Tab
                with stress_tabs[0]:
                    st.write("Normal stress due to bending:")

                    # Create a Plotly figure for normal stress
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["normal_stress_top"] / 1e6,  # Convert to MPa
                        mode='lines',
                        name='Top Fiber (Compression)'
                    ))

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["normal_stress_bottom"] / 1e6,  # Convert to MPa
                        mode='lines',
                        name='Bottom Fiber (Tension)'
                    ))

                    fig.update_layout(
                        title="Normal Stress Distribution",
                        xaxis_title="Position (m)",
                        yaxis_title="Stress (MPa)",
                        legend_title="Location",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Maximum normal stress values
                    col1, col2 = st.columns(2)
                    with col1:
                        max_compression = max(abs(stress_results["normal_stress_top"]))
                        st.metric("Max Compression Stress", f"{max_compression / 1e6:.2f} MPa")

                    with col2:
                        max_tension = max(abs(stress_results["normal_stress_bottom"]))
                        st.metric("Max Tension Stress", f"{max_tension / 1e6:.2f} MPa")

                # Shear Stress Tab
                with stress_tabs[1]:
                    st.write("Shear stress distribution:")

                    # Create a Plotly figure for shear stress
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["max_shear_stress"] / 1e6,  # Convert to MPa
                        mode='lines',
                        name='Max Shear Stress'
                    ))

                    fig.update_layout(
                        title="Shear Stress Distribution",
                        xaxis_title="Position (m)",
                        yaxis_title="Stress (MPa)",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Maximum shear stress value
                    max_shear_stress = max(abs(stress_results["max_shear_stress"]))
                    st.metric("Maximum Shear Stress", f"{max_shear_stress / 1e6:.2f} MPa")

                # Safety Factors Tab
                with stress_tabs[2]:
                    st.write(
                        f"Safety factors based on {material_name} yield strength of {stress_results['yield_strength'] / 1e6:.1f} MPa:")

                    # Create a Plotly figure for safety factors
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["safety_factor_compression"],
                        mode='lines',
                        name='Compression'
                    ))

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["safety_factor_tension"],
                        mode='lines',
                        name='Tension'
                    ))

                    fig.add_trace(go.Scatter(
                        x=stress_results["x"],
                        y=stress_results["safety_factor_shear"],
                        mode='lines',
                        name='Shear'
                    ))

                    fig.update_layout(
                        title="Safety Factor Distribution",
                        xaxis_title="Position (m)",
                        yaxis_title="Safety Factor",
                        legend_title="Stress Type",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Minimum safety factor and critical locations
                    min_sf = min(stress_results["min_safety_factor"])
                    st.metric("Minimum Safety Factor", f"{min_sf:.2f}")

                    st.write(f"Critical location for normal stress: {stress_results['critical_x_normal']:.2f} m")
                    st.write(f"Critical location for shear stress: {stress_results['critical_x_shear']:.2f} m")

                    # Design check
                    if min_sf < 1.0:
                        st.error("UNSAFE DESIGN: Safety factor is less than 1.0!")
                    elif min_sf < 1.5:
                        st.warning("MARGINAL DESIGN: Safety factor is less than recommended minimum of 1.5!")
                    else:
                        st.success("SAFE DESIGN: Safety factor exceeds minimum recommendations!")

            # Display maximum values
            st.subheader("Maximum Values")
            col1, col2, col3 = st.columns(3)

            with col1:
                max_shear = max(abs(min(st.session_state.results["shear_force"])),
                                abs(max(st.session_state.results["shear_force"])))
                st.metric("Maximum Shear Force", f"{max_shear:.2f} N")

            with col2:
                max_moment = max(abs(min(st.session_state.results["bending_moment"])),
                                 abs(max(st.session_state.results["bending_moment"])))
                st.metric("Maximum Bending Moment", f"{max_moment:.2f} N·m")

            with col3:
                max_deflection = max(abs(min(st.session_state.results["deflection"])),
                                     abs(max(st.session_state.results["deflection"])))
                st.metric("Maximum Deflection", f"{max_deflection * 1000:.2f} mm")

            # Display equations
            st.subheader("Governing Equations")
            equations = get_beam_equations(
                st.session_state.beam_type,
                st.session_state.beam_length,
                st.session_state.loads
            )

            for eq_name, eq_latex in equations.items():
                st.markdown(f"**{eq_name}:** {eq_latex}")
        else:
            st.info("Calculate the results first in the 'Load Definition' tab.")

    # Tab 3: Report Generation
    with tab3:
        st.header("Export & Report Generation")

        if st.session_state.show_results and st.session_state.results is not None:
            # Create tabs for different export options
            export_tab1, export_tab2 = st.tabs(["PDF Report", "CSV Data Export"])

            # Tab 1: PDF Report Generation
            with export_tab1:
                st.subheader("PDF Report Options")

                col1, col2 = st.columns(2)
                with col1:
                    include_diagrams = st.checkbox("Include Diagrams", value=True)
                    include_equations = st.checkbox("Include Equations", value=True)

                with col2:
                    include_max_values = st.checkbox("Include Maximum Values", value=True)
                    include_reactions = st.checkbox("Include Support Reactions", value=True)

                project_name = st.text_input("Project Name", "Beam Analysis Project")
                engineer_name = st.text_input("Engineer Name", "")

                # Generate report button
                if st.button("Generate PDF Report"):
                    try:
                        report_bytes = generate_report(
                            project_name=project_name,
                            engineer_name=engineer_name,
                            beam_type=st.session_state.beam_type,
                            beam_length=st.session_state.beam_length,
                            loads=st.session_state.loads,
                            results=st.session_state.results,
                            include_diagrams=include_diagrams,
                            include_equations=include_equations,
                            include_max_values=include_max_values,
                            include_reactions=include_reactions,
                            elastic_modulus=st.session_state.elastic_modulus,
                            moment_of_inertia=st.session_state.moment_of_inertia
                        )

                        # Create download link
                        b64_pdf = base64.b64encode(report_bytes).decode('utf-8')
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{project_name.replace(" ", "_")}_report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("PDF Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

            # Tab 2: CSV Data Export
            with export_tab2:
                st.subheader("CSV Data Export Options")

                # Select which data to export
                export_options = st.multiselect(
                    "Select data to export:",
                    ["Beam Properties", "Loads", "Support Reactions", "Shear Force", "Bending Moment", "Deflection"],
                    default=["Shear Force", "Bending Moment", "Deflection"]
                )

                # Export button
                if st.button("Generate CSV Export"):
                    try:
                        # Create a buffer for CSV output
                        csv_buffer = io.StringIO()

                        # Write beam properties if selected
                        if "Beam Properties" in export_options:
                            csv_buffer.write("# Beam Properties\n")
                            csv_buffer.write(f"Beam Type,{st.session_state.beam_type}\n")
                            csv_buffer.write(f"Beam Length (m),{st.session_state.beam_length}\n")
                            csv_buffer.write(f"Elastic Modulus (N/m²),{st.session_state.elastic_modulus}\n")
                            csv_buffer.write(f"Moment of Inertia (m⁴),{st.session_state.moment_of_inertia}\n\n")

                        # Write loads if selected
                        if "Loads" in export_options and st.session_state.loads:
                            csv_buffer.write("# Loads\n")
                            for i, load in enumerate(st.session_state.loads):
                                load_type = load["type"]
                                if load_type == "Point Load":
                                    csv_buffer.write(
                                        f"Load {i + 1},Point Load,Position (m),{load['position']},Magnitude (N),{load['magnitude']}\n")
                                elif load_type == "UDL":
                                    csv_buffer.write(
                                        f"Load {i + 1},UDL,Start (m),{load['start']},End (m),{load['end']},Magnitude (N/m),{load['magnitude']}\n")
                                elif load_type == "Linearly Varying":
                                    csv_buffer.write(
                                        f"Load {i + 1},Linearly Varying,Start (m),{load['start']},End (m),{load['end']},Start Magnitude (N/m),{load['start_magnitude']},End Magnitude (N/m),{load['end_magnitude']}\n")
                                elif load_type == "Moment":
                                    csv_buffer.write(
                                        f"Load {i + 1},Moment,Position (m),{load['position']},Magnitude (N·m),{load['magnitude']}\n")
                            csv_buffer.write("\n")

                        # Write support reactions if selected
                        if "Support Reactions" in export_options:
                            csv_buffer.write("# Support Reactions\n")
                            for support, value in st.session_state.results["reactions"].items():
                                csv_buffer.write(f"{support},{value}\n")
                            csv_buffer.write("\n")

                        # Create a DataFrame for numerical results
                        results_data = {
                            "Position (m)": st.session_state.results["x"]
                        }

                        if "Shear Force" in export_options:
                            results_data["Shear Force (N)"] = st.session_state.results["shear_force"]

                        if "Bending Moment" in export_options:
                            results_data["Bending Moment (N·m)"] = st.session_state.results["bending_moment"]

                        if "Deflection" in export_options:
                            results_data["Deflection (m)"] = st.session_state.results["deflection"]
                            results_data["Deflection (mm)"] = st.session_state.results["deflection"] * 1000

                        # Add results data if any numerical data is selected
                        if "Shear Force" in export_options or "Bending Moment" in export_options or "Deflection" in export_options:
                            csv_buffer.write("# Results\n")
                            results_df = pd.DataFrame(results_data)
                            csv_buffer.write(results_df.to_csv(index=False))

                        # Convert to bytes and create download link
                        csv_string = csv_buffer.getvalue()
                        b64_csv = base64.b64encode(csv_string.encode()).decode()
                        filename = f"{project_name.replace(' ', '_')}_data.csv"
                        href = f'<a href="data:text/csv;base64,{b64_csv}" download="{filename}">Download CSV Data</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("CSV data export generated successfully!")

                    except Exception as e:
                        st.error(f"Error generating CSV export: {str(e)}")
        else:
            st.info("Calculate the results first in the 'Load Definition' tab.")

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Structural Beam Analysis Tool | Created with Streamlit by JW")


if __name__ == "__main__":
    main()
