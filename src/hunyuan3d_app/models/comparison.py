"""Model comparison and benchmarking tools"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

import gradio as gr
import numpy as np
import torch
from PIL import Image

from .gguf import GGUFModelManager, GGUFModelInfo

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a model benchmark"""
    model_name: str
    model_type: str
    quantization: Optional[str]
    test_name: str
    
    # Performance metrics
    load_time: float
    avg_generation_time: float
    steps_per_second: float
    memory_used_gb: float
    memory_peak_gb: float
    
    # Quality metrics
    quality_score: float  # 0-1, subjective or computed
    artifacts_score: float  # 0-1, lower is better
    consistency_score: float  # 0-1, higher is better
    
    # Test details
    test_prompts: List[str]
    test_params: Dict[str, Any]
    output_samples: List[str]  # Paths to sample outputs
    
    # Metadata
    timestamp: float
    gpu_model: str
    cuda_version: str
    system_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary"""
        return cls(**data)


class ModelComparison:
    """Tools for comparing and benchmarking models"""
    
    DEFAULT_TEST_PROMPTS = [
        "A majestic mountain landscape at golden hour with dramatic clouds",
        "A futuristic cyberpunk city street at night with neon lights",
        "A detailed portrait of an elderly wizard with a long white beard",
        "A steampunk mechanical dragon with intricate gears and brass details",
        "A serene Japanese garden with cherry blossoms and a koi pond"
    ]
    
    def __init__(self, output_dir: Path, cache_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.cache_dir / "benchmark_results.json"
        self.comparison_cache = self._load_results_cache()
        
    def _load_results_cache(self) -> Dict[str, BenchmarkResult]:
        """Load cached benchmark results"""
        cache = {}
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        cache[key] = BenchmarkResult.from_dict(value)
            except Exception as e:
                logger.error(f"Error loading results cache: {e}")
        return cache
        
    def _save_results_cache(self):
        """Save benchmark results to cache"""
        try:
            data = {
                key: result.to_dict() 
                for key, result in self.comparison_cache.items()
            }
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving results cache: {e}")
            
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark"""
        info = {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
            
        return info
        
    def benchmark_model(
        self,
        model,  # The actual model/pipeline object
        model_name: str,
        model_type: str,
        quantization: Optional[str] = None,
        test_prompts: Optional[List[str]] = None,
        test_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> BenchmarkResult:
        """Benchmark a single model
        
        Args:
            model: The model/pipeline to benchmark
            model_name: Name of the model
            model_type: Type of model (image, 3d, etc.)
            quantization: Quantization level if applicable
            test_prompts: Prompts to test with
            test_params: Generation parameters
            progress_callback: Progress callback function
            
        Returns:
            Benchmark result
        """
        if test_prompts is None:
            test_prompts = self.DEFAULT_TEST_PROMPTS[:3]  # Use fewer for speed
            
        if test_params is None:
            test_params = {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
            
        logger.info(f"Starting benchmark for {model_name}")
        
        # System info
        system_info = self._get_system_info()
        
        # Track metrics
        load_start = time.time()
        
        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # Memory before
        memory_before = 0
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            
        # Generation tests
        generation_times = []
        output_samples = []
        quality_scores = []
        
        for i, prompt in enumerate(test_prompts):
            if progress_callback:
                progress_callback(
                    (i + 1) / len(test_prompts),
                    f"Testing prompt {i + 1}/{len(test_prompts)}"
                )
                
            try:
                # Time generation
                gen_start = time.time()
                
                # Generate image
                with torch.no_grad():
                    result = model(
                        prompt=prompt,
                        negative_prompt="blurry, low quality",
                        **test_params
                    )
                    
                gen_time = time.time() - gen_start
                generation_times.append(gen_time)
                
                # Save sample output
                if hasattr(result, 'images') and result.images:
                    image = result.images[0]
                    sample_path = self.output_dir / f"{model_name}_{quantization or 'fp16'}_sample_{i}.png"
                    image.save(sample_path)
                    output_samples.append(str(sample_path))
                    
                    # Simple quality assessment (could be more sophisticated)
                    quality = self._assess_image_quality(image)
                    quality_scores.append(quality)
                    
            except Exception as e:
                logger.error(f"Error during generation test: {e}")
                generation_times.append(float('inf'))
                quality_scores.append(0.0)
                
        # Calculate metrics
        load_time = generation_times[0] if generation_times else 0  # First gen includes loading
        avg_generation_time = np.mean(generation_times[1:]) if len(generation_times) > 1 else generation_times[0]
        steps_per_second = test_params["num_inference_steps"] / avg_generation_time if avg_generation_time > 0 else 0
        
        # Memory metrics
        memory_after = 0
        memory_peak = 0
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_peak = torch.cuda.max_memory_allocated() / (1024**3)
            
        memory_used = memory_after - memory_before
        
        # Quality metrics
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            model_type=model_type,
            quantization=quantization,
            test_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            load_time=load_time,
            avg_generation_time=avg_generation_time,
            steps_per_second=steps_per_second,
            memory_used_gb=memory_used,
            memory_peak_gb=memory_peak,
            quality_score=avg_quality,
            artifacts_score=0.1,  # Placeholder
            consistency_score=0.9,  # Placeholder
            test_prompts=test_prompts,
            test_params=test_params,
            output_samples=output_samples,
            timestamp=time.time(),
            gpu_model=system_info.get("gpu_model", "CPU"),
            cuda_version=system_info.get("cuda_version", "N/A"),
            system_info=system_info
        )
        
        # Cache result
        cache_key = f"{model_name}_{quantization or 'fp16'}_{result.test_name}"
        self.comparison_cache[cache_key] = result
        self._save_results_cache()
        
        logger.info(f"Benchmark complete for {model_name}")
        return result
        
    def _assess_image_quality(self, image: Image.Image) -> float:
        """Simple image quality assessment
        
        Args:
            image: PIL Image
            
        Returns:
            Quality score 0-1
        """
        # This is a placeholder - could use more sophisticated metrics
        # like BRISQUE, NIQE, or perceptual quality metrics
        
        # For now, check basic properties
        width, height = image.size
        
        # Resolution score
        resolution_score = min(width * height / (1024 * 1024), 1.0)
        
        # Color distribution score (entropy)
        histogram = image.histogram()
        entropy = 0
        total_pixels = width * height
        
        for count in histogram:
            if count > 0:
                probability = count / total_pixels
                entropy -= probability * np.log2(probability + 1e-10)
                
        entropy_score = min(entropy / 8.0, 1.0)  # Normalize to 0-1
        
        # Combine scores
        quality_score = (resolution_score + entropy_score) / 2
        
        return quality_score
        
    def compare_models(
        self,
        models: List[Tuple[Any, str, str, Optional[str]]],  # (model, name, type, quantization)
        test_prompts: Optional[List[str]] = None,
        test_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[BenchmarkResult]:
        """Compare multiple models
        
        Args:
            models: List of (model, name, type, quantization) tuples
            test_prompts: Test prompts
            test_params: Generation parameters
            progress_callback: Progress callback
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for i, (model, name, model_type, quantization) in enumerate(models):
            if progress_callback:
                progress_callback(
                    i / len(models),
                    f"Benchmarking {name} ({i + 1}/{len(models)})"
                )
                
            result = self.benchmark_model(
                model=model,
                model_name=name,
                model_type=model_type,
                quantization=quantization,
                test_prompts=test_prompts,
                test_params=test_params
            )
            
            results.append(result)
            
            # Clean up between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    def generate_comparison_report(
        self,
        results: List[BenchmarkResult],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a comparison report
        
        Args:
            results: Benchmark results to compare
            output_path: Optional path to save report
            
        Returns:
            HTML report content
        """
        # Sort by performance
        results_by_speed = sorted(results, key=lambda r: r.steps_per_second, reverse=True)
        results_by_quality = sorted(results, key=lambda r: r.quality_score, reverse=True)
        results_by_memory = sorted(results, key=lambda r: r.memory_used_gb)
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
                h1, h2 {{ color: #333; }}
                .metric {{ font-weight: bold; }}
                .samples {{ display: flex; gap: 10px; flex-wrap: wrap; }}
                .sample img {{ max-width: 200px; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Quantization</th>
                    <th>Speed (steps/s)</th>
                    <th>Memory (GB)</th>
                    <th>Quality Score</th>
                    <th>Load Time (s)</th>
                </tr>
        """
        
        for result in results:
            # Highlight best/worst
            speed_class = "best" if result == results_by_speed[0] else ""
            quality_class = "best" if result == results_by_quality[0] else ""
            memory_class = "best" if result == results_by_memory[0] else ""
            
            html += f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td>{result.quantization or 'FP16'}</td>
                    <td class="{speed_class}">{result.steps_per_second:.2f}</td>
                    <td class="{memory_class}">{result.memory_used_gb:.2f}</td>
                    <td class="{quality_class}">{result.quality_score:.2%}</td>
                    <td>{result.load_time:.1f}</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>Detailed Results</h2>
        """
        
        for result in results:
            html += f"""
            <h3>{result.model_name} ({result.quantization or 'FP16'})</h3>
            <div class="details">
                <p><span class="metric">Average Generation Time:</span> {result.avg_generation_time:.2f}s</p>
                <p><span class="metric">Peak Memory Usage:</span> {result.memory_peak_gb:.2f}GB</p>
                <p><span class="metric">GPU:</span> {result.gpu_model}</p>
                
                <h4>Sample Outputs</h4>
                <div class="samples">
            """
            
            for i, (prompt, sample_path) in enumerate(zip(result.test_prompts, result.output_samples)):
                if Path(sample_path).exists():
                    html += f"""
                    <div class="sample">
                        <img src="{sample_path}" alt="Sample {i + 1}">
                        <p>{prompt[:50]}...</p>
                    </div>
                    """
                    
            html += """
                </div>
            </div>
            """
            
        html += """
            <h2>Recommendations</h2>
            <ul>
                <li><strong>Best Performance:</strong> {best_speed}</li>
                <li><strong>Best Quality:</strong> {best_quality}</li>
                <li><strong>Best Memory Efficiency:</strong> {best_memory}</li>
                <li><strong>Best Overall:</strong> {best_overall}</li>
            </ul>
        </body>
        </html>
        """.format(
            best_speed=f"{results_by_speed[0].model_name} ({results_by_speed[0].quantization or 'FP16'})",
            best_quality=f"{results_by_quality[0].model_name} ({results_by_quality[0].quantization or 'FP16'})",
            best_memory=f"{results_by_memory[0].model_name} ({results_by_memory[0].quantization or 'FP16'})",
            best_overall=self._get_best_overall(results)
        )
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html)
                
        return html
        
    def _get_best_overall(self, results: List[BenchmarkResult]) -> str:
        """Determine best overall model
        
        Args:
            results: Benchmark results
            
        Returns:
            Best model description
        """
        # Simple weighted scoring
        scores = {}
        
        for result in results:
            # Normalize metrics
            speed_score = result.steps_per_second / max(r.steps_per_second for r in results)
            quality_score = result.quality_score
            memory_score = 1 - (result.memory_used_gb / max(r.memory_used_gb for r in results))
            
            # Weighted average (adjust weights as needed)
            overall_score = (
                speed_score * 0.3 +
                quality_score * 0.5 +
                memory_score * 0.2
            )
            
            scores[result] = overall_score
            
        best_result = max(scores.items(), key=lambda x: x[1])[0]
        return f"{best_result.model_name} ({best_result.quantization or 'FP16'})"
        
    def create_ui_component(self) -> gr.Group:
        """Create Gradio UI component for model comparison
        
        Returns:
            Gradio Group component
        """
        with gr.Group() as comparison_group:
            gr.Markdown("### 🔬 Model Comparison & Benchmarking")
            
            # Model selection
            with gr.Row():
                model_choices = gr.CheckboxGroup(
                    choices=[
                        "FLUX.1-dev Q8",
                        "FLUX.1-dev Q6",
                        "FLUX.1-dev Q4",
                        "FLUX.1-schnell Q8",
                        "SDXL Turbo"
                    ],
                    value=["FLUX.1-dev Q8", "FLUX.1-dev Q4"],
                    label="Select models to compare"
                )
                
            # Test configuration
            with gr.Row():
                with gr.Column():
                    test_prompts = gr.Textbox(
                        label="Test prompts (one per line)",
                        lines=5,
                        value="\n".join(self.DEFAULT_TEST_PROMPTS[:3])
                    )
                    
                with gr.Column():
                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Inference steps"
                    )
                    
                    resolution = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=128,
                        label="Resolution"
                    )
                    
            # Actions
            with gr.Row():
                run_comparison_btn = gr.Button(
                    "🚀 Run Comparison",
                    variant="primary"
                )
                
                export_report_btn = gr.Button(
                    "📊 Export Report"
                )
                
            # Results display
            comparison_results = gr.HTML(
                label="Comparison Results"
            )
            
            # Progress
            progress = gr.Progress()
            
            # Handler
            def run_comparison(models, prompts, steps, resolution):
                """Run model comparison"""
                # This would need actual model loading logic
                results_html = """
                <div style="padding: 1rem;">
                    <h4>Comparison Results</h4>
                    <p>Model comparison would run here with selected models.</p>
                    <p>This is a placeholder for the actual implementation.</p>
                </div>
                """
                
                return results_html
                
            run_comparison_btn.click(
                run_comparison,
                inputs=[model_choices, test_prompts, steps, resolution],
                outputs=[comparison_results]
            )
            
        return comparison_group
        

import sys