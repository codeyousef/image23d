"""Statistics and analytics for generation history"""

import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .models import GenerationRecord


class StatisticsManager:
    """Manages statistics and analytics for generation history"""
    
    @staticmethod
    def calculate_statistics(records: List[GenerationRecord]) -> Dict[str, Any]:
        """Calculate statistics from generation records
        
        Args:
            records: List of generation records
            
        Returns:
            Dictionary of statistics
        """
        if not records:
            return {
                "total": 0,
                "by_type": {},
                "by_model": {},
                "favorites": 0,
                "unviewed": 0,
                "time_range": {
                    "earliest": None,
                    "latest": None,
                    "span_days": 0
                },
                "generation_rate": {
                    "last_hour": 0,
                    "last_day": 0,
                    "last_week": 0,
                    "last_month": 0
                },
                "popular_tags": [],
                "avg_parameters": {}
            }
        
        # Basic counts
        stats = {
            "total": len(records),
            "favorites": sum(1 for r in records if r.favorite),
            "unviewed": sum(1 for r in records if not r.viewed)
        }
        
        # Group by type
        type_counter = Counter(r.generation_type for r in records)
        stats["by_type"] = dict(type_counter)
        
        # Group by model
        model_counter = Counter(r.model_name for r in records)
        stats["by_model"] = dict(model_counter)
        
        # Time range
        timestamps = [r.timestamp for r in records]
        stats["time_range"] = {
            "earliest": min(timestamps),
            "latest": max(timestamps),
            "span_days": (max(timestamps) - min(timestamps)) / (24 * 60 * 60)
        }
        
        # Generation rate
        current_time = time.time()
        stats["generation_rate"] = {
            "last_hour": sum(1 for r in records if current_time - r.timestamp < 3600),
            "last_day": sum(1 for r in records if current_time - r.timestamp < 86400),
            "last_week": sum(1 for r in records if current_time - r.timestamp < 604800),
            "last_month": sum(1 for r in records if current_time - r.timestamp < 2592000)
        }
        
        # Popular tags
        all_tags = []
        for r in records:
            if r.tags:
                all_tags.extend(r.tags)
        tag_counter = Counter(all_tags)
        stats["popular_tags"] = tag_counter.most_common(10)
        
        # Average parameters by type
        stats["avg_parameters"] = StatisticsManager._calculate_avg_parameters(records)
        
        return stats
    
    @staticmethod
    def _calculate_avg_parameters(records: List[GenerationRecord]) -> Dict[str, Dict[str, float]]:
        """Calculate average parameters by generation type"""
        params_by_type = {}
        
        for record in records:
            gen_type = record.generation_type
            if gen_type not in params_by_type:
                params_by_type[gen_type] = {
                    "counts": {},
                    "sums": {}
                }
            
            # Accumulate numeric parameters
            for key, value in record.parameters.items():
                if isinstance(value, (int, float)):
                    if key not in params_by_type[gen_type]["sums"]:
                        params_by_type[gen_type]["sums"][key] = 0
                        params_by_type[gen_type]["counts"][key] = 0
                    
                    params_by_type[gen_type]["sums"][key] += value
                    params_by_type[gen_type]["counts"][key] += 1
        
        # Calculate averages
        avg_params = {}
        for gen_type, data in params_by_type.items():
            avg_params[gen_type] = {}
            for key, total in data["sums"].items():
                count = data["counts"][key]
                if count > 0:
                    avg_params[gen_type][key] = round(total / count, 2)
        
        return avg_params
    
    @staticmethod
    def generate_timeline(records: List[GenerationRecord], days: int = 30) -> Dict[str, List[int]]:
        """Generate timeline data for visualization
        
        Args:
            records: List of generation records
            days: Number of days to include
            
        Returns:
            Dictionary with daily counts by type
        """
        # Initialize timeline
        timeline = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        # Initialize counts
        types = set(r.generation_type for r in records)
        for gen_type in types:
            timeline[gen_type] = [0] * len(date_range)
        
        # Count generations per day
        for record in records:
            record_date = datetime.fromtimestamp(record.timestamp)
            if record_date >= start_date:
                date_str = record_date.strftime("%Y-%m-%d")
                if date_str in date_range:
                    day_index = date_range.index(date_str)
                    if record.generation_type in timeline:
                        timeline[record.generation_type][day_index] += 1
        
        # Add dates to result
        timeline["dates"] = date_range
        
        return timeline
    
    @staticmethod
    def get_usage_patterns(records: List[GenerationRecord]) -> Dict[str, Any]:
        """Analyze usage patterns
        
        Args:
            records: List of generation records
            
        Returns:
            Dictionary of usage patterns
        """
        if not records:
            return {
                "peak_hours": [],
                "peak_days": [],
                "avg_daily_generations": 0,
                "most_productive_period": None
            }
        
        # Hour of day analysis
        hour_counter = Counter()
        day_counter = Counter()
        
        for record in records:
            dt = datetime.fromtimestamp(record.timestamp)
            hour_counter[dt.hour] += 1
            day_counter[dt.strftime("%A")] += 1
        
        # Peak hours (top 3)
        peak_hours = [(hour, count) for hour, count in hour_counter.most_common(3)]
        
        # Peak days
        peak_days = [(day, count) for day, count in day_counter.most_common(3)]
        
        # Average daily generations
        time_span_days = (max(r.timestamp for r in records) - min(r.timestamp for r in records)) / 86400
        avg_daily = len(records) / max(time_span_days, 1)
        
        # Most productive period (find highest concentration)
        most_productive = StatisticsManager._find_most_productive_period(records)
        
        return {
            "peak_hours": peak_hours,
            "peak_days": peak_days,
            "avg_daily_generations": round(avg_daily, 2),
            "most_productive_period": most_productive
        }
    
    @staticmethod
    def _find_most_productive_period(
        records: List[GenerationRecord], 
        window_hours: int = 4
    ) -> Optional[Dict[str, Any]]:
        """Find the most productive time period"""
        if not records:
            return None
        
        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        
        max_count = 0
        max_period = None
        window_seconds = window_hours * 3600
        
        for i, record in enumerate(sorted_records):
            # Count records within window
            count = 1
            end_time = record.timestamp + window_seconds
            
            for j in range(i + 1, len(sorted_records)):
                if sorted_records[j].timestamp <= end_time:
                    count += 1
                else:
                    break
            
            if count > max_count:
                max_count = count
                max_period = {
                    "start": datetime.fromtimestamp(record.timestamp),
                    "end": datetime.fromtimestamp(min(end_time, sorted_records[i + count - 1].timestamp)),
                    "count": count,
                    "rate_per_hour": count / window_hours
                }
        
        return max_period