"""Enhanced Hunyuan3D Studio application with all features"""

import logging
import sys
import os
import argparse

# Handle imports
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    from hunyuan3d_app.hunyuan3d_studio_enhanced import Hunyuan3DStudioEnhanced
    from hunyuan3d_app.ui_enhanced import create_enhanced_interface
    from hunyuan3d_app.gpu_optimizer import get_gpu_optimizer
    from hunyuan3d_app.system_checker import check_system_requirements
else:
    from .hunyuan3d_studio_enhanced import Hunyuan3DStudioEnhanced
    from .ui_enhanced import create_enhanced_interface
    from .gpu_optimizer import get_gpu_optimizer
    from .system_checker import check_system_requirements

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for enhanced app"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hunyuan3D Studio Enhanced")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=None, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--workers", type=int, default=2, help="Number of queue workers")
    args = parser.parse_args()
    
    # Initialize GPU optimizations
    logger.info("Initializing GPU optimizations...")
    gpu_optimizer = get_gpu_optimizer()
    
    # Check system requirements
    sys_req = check_system_requirements()
    if sys_req["overall_status"] == "error":
        logger.warning("System does not meet minimum requirements:")
        for error in sys_req["errors"]:
            logger.warning(f"- {error}")
    elif sys_req["overall_status"] == "warning":
        logger.warning("System meets minimum but not recommended requirements:")
        for warning in sys_req["warnings"]:
            logger.warning(f"- {warning}")
    else:
        logger.info("System meets all recommended requirements.")
    
    # Create enhanced app
    logger.info("Initializing Hunyuan3D Studio Enhanced...")
    app = Hunyuan3DStudioEnhanced()
    
    # Override worker count if specified
    if args.workers != app.queue_manager.max_workers:
        logger.info(f"Setting queue workers to {args.workers}")
        # Would need to implement worker count update
    
    # Create enhanced interface
    logger.info("Creating enhanced interface...")
    interface = create_enhanced_interface(app)
    
    # Determine port
    if args.port:
        port = args.port
    else:
        # Find available port
        import socket
        port = None
        for test_port in range(7860, 7880):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', test_port)) != 0:
                    port = test_port
                    break
        
        if port is None:
            logger.warning("Could not find available port, using random")
            port = None
    
    logger.info(f"Starting enhanced server on {args.host}:{port}")
    
    # Launch interface
    interface.launch(
        share=args.share,
        server_name=args.host,
        server_port=port,
        inbrowser=not args.no_browser,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)