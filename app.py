#!/usr/bin/env python3
"""
PickleTrack Dashboard - Streamlit Cloud Entry Point
Main entry point for Streamlit Cloud deployment
"""

import sys
from pathlib import Path

# Add src to Python path for Streamlit Cloud
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Import and run the dashboard
from streamlit_dashboard import PickleTrackStreamlitDashboard

def main():
    """Main entry point for Streamlit Cloud."""
    dashboard = PickleTrackStreamlitDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()