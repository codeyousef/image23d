"""Pipeline validation utilities for HunYuan3D

This module provides validation and debugging utilities to ensure
the HunYuan3D pipeline remains available throughout its lifecycle.
"""

import logging
from typing import Any, Optional, Dict, List
from datetime import datetime
import traceback
import weakref

logger = logging.getLogger(__name__)


class PipelineStateTracker:
    """Tracks pipeline state throughout its lifecycle"""
    
    def __init__(self):
        self.states: List[Dict[str, Any]] = []
        self.pipeline_refs: List[weakref.ref] = []
        
    def log_state(self, component: str, pipeline: Any, context: Dict[str, Any] = None):
        """Log the current state of the pipeline"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'pipeline_id': id(pipeline) if pipeline else None,
            'pipeline_type': type(pipeline).__name__ if pipeline else 'None',
            'is_none': pipeline is None,
            'context': context or {},
            'stack_trace': traceback.extract_stack()[-5:-1]  # Last 4 stack frames
        }
        
        # Add weak reference if pipeline exists
        if pipeline is not None:
            self.pipeline_refs.append(weakref.ref(pipeline))
            
        self.states.append(state)
        
        # Log immediately
        logger.info(f"Pipeline state at {component}: "
                   f"id={state['pipeline_id']}, "
                   f"type={state['pipeline_type']}, "
                   f"is_none={state['is_none']}")
        
    def validate_consistency(self) -> bool:
        """Validate that pipeline references are consistent"""
        if not self.states:
            return True
            
        # Get all non-None pipeline IDs
        pipeline_ids = [s['pipeline_id'] for s in self.states if s['pipeline_id'] is not None]
        
        if not pipeline_ids:
            logger.warning("No valid pipeline IDs found in state history")
            return False
            
        # Check if all IDs are the same
        unique_ids = set(pipeline_ids)
        if len(unique_ids) > 1:
            logger.error(f"Multiple pipeline IDs detected: {unique_ids}")
            self._log_state_history()
            return False
            
        return True
        
    def _log_state_history(self):
        """Log the full state history for debugging"""
        logger.error("=== Pipeline State History ===")
        for i, state in enumerate(self.states):
            logger.error(f"State {i}: {state['timestamp']} - {state['component']}")
            logger.error(f"  Pipeline: id={state['pipeline_id']}, type={state['pipeline_type']}")
            logger.error(f"  Stack: {[f'{frame.filename}:{frame.lineno}' for frame in state['stack_trace']]}")
            

class PipelineValidator:
    """Validates HunYuan3D pipeline state and provides recovery mechanisms"""
    
    def __init__(self):
        self.state_tracker = PipelineStateTracker()
        self.validation_enabled = True
        
    def validate_pipeline(self, pipeline: Any, component: str, context: Dict[str, Any] = None) -> bool:
        """Validate pipeline state at a specific point"""
        if not self.validation_enabled:
            return True
            
        # Track state
        self.state_tracker.log_state(component, pipeline, context)
        
        # Perform validation
        if pipeline is None:
            logger.error(f"Pipeline is None at {component}")
            self._log_debug_info(component, context)
            return False
            
        # Check if it's the expected type
        expected_type = "Hunyuan3DDiTFlowMatchingPipeline"
        actual_type = type(pipeline).__name__
        
        if expected_type not in actual_type and actual_type != 'MagicMock':  # Allow mocks in tests
            logger.warning(f"Unexpected pipeline type at {component}: {actual_type}")
            
        # Validate consistency
        if not self.state_tracker.validate_consistency():
            logger.error(f"Pipeline consistency check failed at {component}")
            return False
            
        return True
        
    def _log_debug_info(self, component: str, context: Dict[str, Any] = None):
        """Log detailed debug information"""
        logger.error(f"=== Pipeline Validation Failed at {component} ===")
        
        if context:
            logger.error(f"Context: {context}")
            
        # Log recent states
        recent_states = self.state_tracker.states[-5:]
        for state in recent_states:
            logger.error(f"Recent state: {state['component']} - "
                        f"Pipeline ID: {state['pipeline_id']}, "
                        f"Is None: {state['is_none']}")
                        
    def validate_multiview_pipeline(self, multiview_model: Any) -> bool:
        """Specifically validate multiview model pipeline"""
        if not hasattr(multiview_model, 'pipeline'):
            logger.error("Multiview model has no pipeline attribute")
            return False
            
        pipeline = getattr(multiview_model, 'pipeline', None)
        return self.validate_pipeline(
            pipeline, 
            "multiview_model",
            {'model_loaded': getattr(multiview_model, 'loaded', False)}
        )
        
    def validate_reconstruction_access(self, reconstruction_model: Any) -> bool:
        """Validate pipeline access from reconstruction model"""
        if not hasattr(reconstruction_model, 'multiview_model'):
            logger.error("Reconstruction model has no multiview_model attribute")
            return False
            
        multiview = getattr(reconstruction_model, 'multiview_model', None)
        if multiview is None:
            logger.error("Reconstruction model's multiview_model is None")
            return False
            
        pipeline = getattr(multiview, 'pipeline', None)
        return self.validate_pipeline(
            pipeline,
            "reconstruction_model.multiview_model",
            {
                'multiview_id': id(multiview),
                'multiview_loaded': getattr(multiview, 'loaded', False)
            }
        )
        
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a comprehensive validation report"""
        states = self.state_tracker.states
        
        report = {
            'total_validations': len(states),
            'none_count': sum(1 for s in states if s['is_none']),
            'unique_pipeline_ids': len(set(s['pipeline_id'] for s in states if s['pipeline_id'])),
            'components_validated': list(set(s['component'] for s in states)),
            'is_consistent': self.state_tracker.validate_consistency()
        }
        
        # Add timeline
        if states:
            report['timeline'] = [
                {
                    'timestamp': s['timestamp'],
                    'component': s['component'],
                    'pipeline_present': not s['is_none']
                }
                for s in states
            ]
            
        return report


# Global validator instance
_global_validator = None


def get_pipeline_validator() -> PipelineValidator:
    """Get the global pipeline validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = PipelineValidator()
    return _global_validator


def validate_pipeline_checkpoint(pipeline: Any, checkpoint_name: str, **context) -> bool:
    """Convenience function to validate pipeline at checkpoints"""
    validator = get_pipeline_validator()
    return validator.validate_pipeline(pipeline, checkpoint_name, context)


def log_pipeline_lifecycle(message: str, pipeline: Any = None, **details):
    """Log pipeline lifecycle events with consistent formatting"""
    log_entry = {
        'message': message,
        'pipeline_id': id(pipeline) if pipeline else 'None',
        'pipeline_type': type(pipeline).__name__ if pipeline else 'None',
        'timestamp': datetime.now().isoformat(),
        **details
    }
    
    logger.info(f"[Pipeline Lifecycle] {message} | "
               f"ID: {log_entry['pipeline_id']} | "
               f"Type: {log_entry['pipeline_type']}")
    
    if details:
        logger.debug(f"[Pipeline Lifecycle] Details: {details}")
        

class PipelineGuard:
    """Context manager to guard pipeline state during operations"""
    
    def __init__(self, pipeline_holder: Any, attribute_name: str = 'pipeline'):
        self.holder = pipeline_holder
        self.attribute_name = attribute_name
        self.original_pipeline = None
        self.pipeline_id = None
        
    def __enter__(self):
        """Store pipeline state on entry"""
        if hasattr(self.holder, self.attribute_name):
            self.original_pipeline = getattr(self.holder, self.attribute_name)
            self.pipeline_id = id(self.original_pipeline) if self.original_pipeline else None
            log_pipeline_lifecycle(
                f"Entering guarded section",
                self.original_pipeline,
                holder_type=type(self.holder).__name__
            )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Verify pipeline state on exit"""
        if hasattr(self.holder, self.attribute_name):
            current_pipeline = getattr(self.holder, self.attribute_name)
            current_id = id(current_pipeline) if current_pipeline else None
            
            if current_id != self.pipeline_id:
                logger.error(f"Pipeline changed during guarded section! "
                           f"Original: {self.pipeline_id}, Current: {current_id}")
                           
            log_pipeline_lifecycle(
                f"Exiting guarded section",
                current_pipeline,
                changed=current_id != self.pipeline_id,
                exception=exc_type.__name__ if exc_type else None
            )