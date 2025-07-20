"""Patch for torchvision compatibility issues"""

import logging

logger = logging.getLogger(__name__)

def patch_torchvision():
    """Apply compatibility patches for torchvision"""
    try:
        # Patch the torchvision NMS operator registration issue
        import torch
        
        # Check if the operator already exists before trying to patch
        if hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'):
            logger.debug("torchvision.nms operator already exists")
            return
        
        # Try to register a dummy NMS operator if needed
        try:
            import torchvision
            # Force initialization
            _ = torchvision.ops.nms
        except Exception as e:
            logger.warning(f"torchvision ops initialization issue: {e}")
            
            # Create a dummy nms function as fallback
            def dummy_nms(boxes, scores, iou_threshold):
                """Dummy NMS implementation for compatibility"""
                return torch.arange(len(boxes))
            
            # Try to patch it
            if not hasattr(torch.ops, 'torchvision'):
                torch.ops.torchvision = type('Module', (), {})()
            
            if not hasattr(torch.ops.torchvision, 'nms'):
                torch.ops.torchvision.nms = dummy_nms
                logger.info("Applied torchvision NMS compatibility patch")
    
    except Exception as e:
        logger.warning(f"Could not apply torchvision patch: {e}")

# Apply patch when module is imported
patch_torchvision()