# Beam Analysis Application

A comprehensive structural engineering tool for analyzing beams under various loading conditions and support configurations.

## Features

- **Multiple Beam Types Support**
  - Cantilever beams
  - Simply supported beams
  - Overhanging beams
  - Fixed-fixed beams

- **Load Analysis**
  - Point loads
  - Distributed loads
  - Moment loads
  - Support for multiple load combinations

- **Structural Analysis**
  - Shear force calculations
  - Bending moment calculations
  - Deflection analysis
  - Stress distribution analysis
  - Reaction force calculations

- **Section Properties**
  - Support for various cross-sections
  - Moment of inertia calculations
  - Section property lookups

- **Design Code Compliance**
  - Multiple design code support
  - Load combination factors
  - Code-specific analysis

- **Visualization**
  - Beam diagrams
  - Shear force diagrams
  - Bending moment diagrams
  - Deflection diagrams
  - Interactive plots

- **Reporting**
  - Comprehensive PDF reports
  - Customizable report content
  - Professional formatting

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - matplotlib
  - sympy
  - reportlab
  - streamlit

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/MAELSTROM001/_BeAn-Tool]
cd BeamAnalysis_SAlone
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. The application will open in your default web browser. Follow the interactive interface to:
   - Select beam type
   - Define beam properties
   - Add loads
   - Choose analysis options
   - Generate reports

## Example

```python
# Example of analyzing a simply supported beam with a point load
beam_type = "Supported-Supported"
beam_length = 5.0  # meters
loads = [
    {
        "type": "Point Load",
        "position": 2.5,
        "magnitude": 1000  # Newtons
    }
]
```

## Output

The application generates:
- Interactive plots of beam response
- Detailed PDF reports
- Numerical results including:
  - Maximum shear force
  - Maximum bending moment
  - Maximum deflection
  - Support reactions
  - Stress distributions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Contact

[Your contact information] 
