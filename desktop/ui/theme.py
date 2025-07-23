"""
NeuralForge Studio theme configuration
"""

from nicegui import ui

# Color palette
COLORS = {
    "primary": "#7C3AED",      # Electric Purple
    "success": "#10B981",      # Emerald
    "background": "#0A0A0A",   # Deep Black
    "surface": "#1F1F1F",      # Card background
    "text": "#E5E5E5",         # Light gray text
    "text_secondary": "#A0A0A0", # Secondary text
    "border": "#333333",       # Border color
    "error": "#EF4444",        # Red
    "warning": "#F59E0B",      # Amber
    "info": "#3B82F6",         # Blue
}

# CSS styles
STYLES = """
/* Global styles */
* {
    transition: all 0.2s ease;
}

body {
    background-color: #0A0A0A !important;
    color: #E5E5E5 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Navigation sidebar */
.navigation-sidebar {
    background-color: #1F1F1F;
    border-right: 1px solid #333333;
    padding: 1rem;
}

.nav-item {
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    border-radius: 0.5rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: #A0A0A0;
    transition: all 0.2s;
}

.nav-item:hover {
    background-color: rgba(124, 58, 237, 0.1);
    color: #E5E5E5;
}

.nav-item.active {
    background-color: rgba(124, 58, 237, 0.2);
    color: #7C3AED;
    font-weight: 500;
}

/* Cards */
.card {
    background-color: #1F1F1F;
    border: 1px solid #333333;
    border-radius: 0.75rem;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Buttons */
.q-btn {
    border-radius: 0.5rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    padding: 0.5rem 1.5rem !important;
}

.q-btn--primary {
    background-color: #7C3AED !important;
}

.q-btn--primary:hover {
    background-color: #6B2FD6 !important;
}

/* Input fields */
.q-field--outlined .q-field__control {
    background-color: #0A0A0A !important;
    border-color: #333333 !important;
}

.q-field--outlined.q-field--focused .q-field__control {
    border-color: #7C3AED !important;
}

/* Progress bars */
.progress-pipeline {
    background-color: #1F1F1F;
    border: 1px solid #333333;
    border-radius: 0.5rem;
    padding: 1rem;
}

.progress-step {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0.375rem;
    background-color: #0A0A0A;
}

.progress-step.active {
    background-color: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.3);
}

.progress-step.completed {
    background-color: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Enhancement fields */
.enhancement-field {
    background-color: #0A0A0A;
    border: 1px solid #333333;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.enhancement-field:hover {
    border-color: #7C3AED;
}

/* Tabs */
.q-tab {
    color: #A0A0A0 !important;
}

.q-tab--active {
    color: #7C3AED !important;
}

.q-tab-panels {
    background-color: transparent !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0A0A0A;
}

::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #7C3AED;
}

/* Animations */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Gallery grid */
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}

.gallery-item {
    background-color: #1F1F1F;
    border: 1px solid #333333;
    border-radius: 0.5rem;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s;
}

.gallery-item:hover {
    transform: scale(1.05);
    border-color: #7C3AED;
}
"""

def apply_theme():
    """Apply the NeuralForge theme to the app"""
    # Set Quasar dark mode
    ui.dark_mode(True)
    
    # Apply custom CSS
    ui.add_head_html(f'<style>{STYLES}</style>')
    
    # Load RTL styles
    from pathlib import Path
    rtl_css_path = Path(__file__).parent / 'styles' / 'rtl.css'
    if rtl_css_path.exists():
        with open(rtl_css_path, 'r') as f:
            rtl_styles = f.read()
        ui.add_head_html(f'<style>{rtl_styles}</style>')
    
    # Configure Quasar colors
    ui.colors(
        primary=COLORS["primary"],
        secondary=COLORS["success"],
        accent=COLORS["primary"],
        dark='#0A0A0A',
        positive=COLORS["success"],
        negative=COLORS["error"],
        info=COLORS["info"],
        warning=COLORS["warning"]
    )