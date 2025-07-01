import logging
import sys
import os

# Handle imports differently when run as a script vs. imported as a module
if __name__ == "__main__":
    # When run directly, use absolute imports
    # Add the parent directory to sys.path to make the package importable
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from hunyuan3d_app.hunyuan3d_studio import Hunyuan3DStudio
    from hunyuan3d_app.ui import create_interface
else:
    # When imported as a module, use relative imports
    from .hunyuan3d_studio import Hunyuan3DStudio
    from .ui import create_interface

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Hunyuan3DStudio()

# Check system requirements
if __name__ == "__main__":
    # When run directly, use absolute imports
    from hunyuan3d_app.system_checker import check_system_requirements
else:
    # When imported as a module, use relative imports
    from .system_checker import check_system_requirements
sys_req = check_system_requirements()
if sys_req["overall_status"] == "error":
    logger.warning("System does not meet minimum requirements:")
    for error in sys_req["errors"]:
        logger.warning(f"- {error}")
    logger.info("See the System Requirements tab for details and recommendations.")
elif sys_req["overall_status"] == "warning":
    logger.warning("System meets minimum but not recommended requirements:")
    for warning in sys_req["warnings"]:
        logger.warning(f"- {warning}")
    logger.info("See the System Requirements tab for details and recommendations.")
else:
    logger.info("System meets all recommended requirements.")

# Create interface
interface = create_interface(app)

if __name__ == "__main__":
    # Get port from environment variable or use a range of ports
    import os
    import socket

    # Try to get port from environment variable
    port = os.environ.get("GRADIO_SERVER_PORT")
    if port:
        port = int(port)
    else:
        # Try ports in range 7860-7870
        port = None
        for test_port in range(7860, 7880):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', test_port)) != 0:
                    # Port is available
                    port = test_port
                    break

        if port is None:
            logger.warning("Could not find an available port in range 7860-7880. Using random port.")

    logger.info(f"Starting server on port {port}")

    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True
    )
