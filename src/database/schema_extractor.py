"""
Schema extraction from MySQL database with report type filtering
"""
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

from .connector import db_connector
from ..utils.config import config
from ..utils.logger import schema_logger, logger

# Report type table mappings
REPORT_TYPE_TABLES = {
    "incident_details": [
        "incidents", "incidents_types", "incident_comments", "fixtures", 
        "business", "media_library", "media_library_file_category", 
        "incidents_locations", "incidents_parties", "incidents_vehicles", 
        "program", "web_constants", "user", "job_titles", "user_program_mapping"
    ],
    "equipment_usage": [
        "equipments", "equipment_activity", "equipment_types", "equipment_upkeep",
        "equipment_category", "user", "program", "business"
    ],
    "all": None  # None means no filter - process all tables
}

class SchemaExtractor:
    """Extracts and processes database schema information with report type filtering"""
    
    def __init__(self):
        self.raw_schema: Dict[str, Any] = {}
        self.processed_schema: Dict[str, Any] = {}
        self.extraction_stats: Dict[str, Any] = {}
        self.report_type: str = "all"
    
    def extract_complete_schema(self, report_type: str = "incident_details") -> Dict[str, Any]:
        """Extract complete database schema with optional report type filtering"""
        schema_logger.log_section("EXTRACTING DATABASE SCHEMA")
        
        self.report_type = report_type
        start_time = datetime.now()
        
        try:
            # Connect to database
            if not db_connector.connect():
                raise RuntimeError("Failed to connect to database")
            
            # Get table filter for report type
            table_filter = REPORT_TYPE_TABLES.get(report_type)
            
            if report_type != "all" and table_filter:
                schema_logger.log_info(f"🎯 Filtering for '{report_type}' report type")
                schema_logger.log_info(f"📋 Target tables: {', '.join(table_filter)}")
                
                # Validate table filter
                validation = db_connector.validate_table_filter(table_filter)
                if validation["missing"]:
                    schema_logger.log_warning(f"⚠️  Tables not found: {', '.join(validation['missing'])}")
                schema_logger.log_info(f"✅ Found {len(validation['existing'])}/{len(table_filter)} target tables")
            else:
                schema_logger.log_info("📊 Processing all database tables")
            
            # Get database info with filter
            schema_logger.log_info("🔍 Extracting database information...")
            db_info = db_connector.get_database_info(table_filter=table_filter)
            
            # Initialize schema structure
            self.raw_schema = {
                "database_info": db_info,
                "report_type": report_type,
                "table_filter": table_filter,
                "tables": {},
                "extraction_metadata": {
                    "extraction_date": start_time.isoformat(),
                    "config": config.to_dict(),
                    "report_type": report_type,
                    "table_filter_applied": bool(table_filter),
                    "total_tables": db_info["total_tables"],
                    "total_columns": 0,
                    "extraction_duration_seconds": 0
                }
            }
            
            # Extract each table
            table_names = db_info["tables"]
            total_tables = len(table_names)
            
            if report_type != "all":
                schema_logger.log_info(f"📊 Processing {total_tables} tables for '{report_type}' report")
            else:
                schema_logger.log_info(f"📊 Processing all {total_tables} tables")
            
            if not table_names:
                schema_logger.log_warning("⚠️  No tables found to process!")
                return self.raw_schema
            
            # Start progress tracking
            progress_task = schema_logger.start_progress("Extracting table schemas...")
            
            total_columns = 0
            processed_tables = 0
            failed_tables = []
            
            for i, table_name in enumerate(table_names):
                try:
                    schema_logger.update_progress(
                        progress_task, 
                        advance=(100 / total_tables),
                        description=f"Processing table: {table_name}"
                    )
                    
                    # Extract table information
                    table_info = db_connector.get_table_info(table_name)
                    self.raw_schema["tables"][table_name] = table_info
                    
                    # Update stats
                    column_count = len(table_info["columns"])
                    total_columns += column_count
                    processed_tables += 1
                    
                    # Log table info
                    row_count = table_info.get("row_count")
                    schema_logger.log_table_info(table_name, column_count, row_count)
                    
                except Exception as e:
                    schema_logger.log_error(f"❌ Failed to extract schema for table {table_name}: {str(e)}")
                    failed_tables.append(table_name)
                    continue
            
            schema_logger.stop_progress()
            
            # Update extraction metadata
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.raw_schema["extraction_metadata"].update({
                "total_columns": total_columns,
                "processed_tables": processed_tables,
                "failed_tables": failed_tables,
                "extraction_duration_seconds": round(duration, 2),
                "extraction_end_date": end_time.isoformat()
            })
            
            # Update stats
            self.extraction_stats = {
                "report_type": report_type,
                "total_tables": total_tables,
                "processed_tables": processed_tables,
                "failed_tables": len(failed_tables),
                "total_columns": total_columns,
                "extraction_duration": duration
            }
            
            # Summary
            schema_logger.log_success(f"🎉 Schema extraction completed successfully!")
            schema_logger.log_info(f"📊 Processed {processed_tables}/{total_tables} tables ({total_columns} columns) in {duration:.2f} seconds")
            
            if report_type != "all":
                schema_logger.log_info(f"🎯 Report type: {report_type}")
            
            if failed_tables:
                schema_logger.log_warning(f"⚠️  Failed tables: {', '.join(failed_tables)}")
            
            return self.raw_schema
            
        except Exception as e:
            schema_logger.log_error(f"💥 Schema extraction failed: {str(e)}")
            raise
        finally:
            db_connector.disconnect()
    
    def get_available_report_types(self) -> Dict[str, List[str]]:
        """Get available report types and their table mappings"""
        return REPORT_TYPE_TABLES.copy()
    
    def validate_report_type(self, report_type: str) -> bool:
        """Validate if report type is supported"""
        return report_type in REPORT_TYPE_TABLES
    
    def get_report_type_info(self, report_type: str) -> Dict[str, Any]:
        """Get information about a specific report type"""
        if report_type not in REPORT_TYPE_TABLES:
            return {"error": f"Unknown report type: {report_type}"}
        
        table_filter = REPORT_TYPE_TABLES[report_type]
        
        if table_filter is None:
            return {
                "report_type": report_type,
                "description": "All database tables",
                "table_count": "All available tables",
                "tables": "All"
            }
        
        return {
            "report_type": report_type,
            "description": f"Tables for {report_type} reporting",
            "table_count": len(table_filter),
            "tables": table_filter
        }
    
    def save_raw_schema(self, output_path: Optional[Path] = None) -> Path:
        """Save raw schema to JSON file"""
        if not self.raw_schema:
            raise ValueError("No schema data to save. Run extract_complete_schema() first.")
        
        if output_path is None:
            # Include report type in filename
            if self.report_type != "all":
                filename = f"raw_schema_{self.report_type}.json"
            else:
                filename = "raw_schema.json"
            output_path = config.get_output_path(filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.raw_schema, 
                    f, 
                    indent=config.processing.json_indent if config.processing.pretty_json else None,
                    ensure_ascii=False,
                    default=str  # Handle non-serializable objects
                )
            
            file_size = output_path.stat().st_size
            schema_logger.log_success(f"💾 Raw schema saved to: {output_path} ({file_size:,} bytes)")
            return output_path
            
        except Exception as e:
            schema_logger.log_error(f"💥 Failed to save raw schema: {str(e)}")
            raise
    
    def get_table_summary(self) -> Dict[str, Any]:
        """Get summary of extracted tables"""
        if not self.raw_schema:
            return {}
        
        summary = {
            "database": self.raw_schema["database_info"]["database_name"],
            "report_type": self.raw_schema.get("report_type", "all"),
            "total_tables": len(self.raw_schema["tables"]),
            "tables": []
        }
        
        for table_name, table_info in self.raw_schema["tables"].items():
            table_summary = {
                "name": table_name,
                "columns": len(table_info["columns"]),
                "primary_keys": len(table_info.get("primary_keys", [])),
                "foreign_keys": len(table_info.get("foreign_keys", [])),
                "indexes": len(table_info.get("indexes", [])),
                "row_count": table_info.get("row_count"),
                "has_sample_data": bool(table_info.get("sample_data"))
            }
            summary["tables"].append(table_summary)
        
        # Sort by column count (descending)
        summary["tables"].sort(key=lambda x: x["columns"], reverse=True)
        
        return summary
    
    def get_column_statistics(self) -> Dict[str, Any]:
        """Get statistics about columns across all tables"""
        if not self.raw_schema:
            return {}
        
        stats = {
            "total_columns": 0,
            "data_types": {},
            "nullable_columns": 0,
            "columns_with_defaults": 0,
            "auto_increment_columns": 0,
            "most_common_column_names": {}
        }
        
        column_names = []
        
        for table_info in self.raw_schema["tables"].values():
            for column in table_info["columns"]:
                stats["total_columns"] += 1
                column_names.append(column["name"])
                
                # Data type statistics
                data_type = column["type"]
                stats["data_types"][data_type] = stats["data_types"].get(data_type, 0) + 1
                
                # Other statistics
                if column.get("nullable", True):
                    stats["nullable_columns"] += 1
                
                if column.get("default") is not None:
                    stats["columns_with_defaults"] += 1
                
                if column.get("autoincrement", False):
                    stats["auto_increment_columns"] += 1
        
        # Most common column names
        from collections import Counter
        name_counts = Counter(column_names)
        stats["most_common_column_names"] = dict(name_counts.most_common(10))
        
        return stats
    
    def generate_schema_report(self) -> str:
        """Generate a human-readable schema report"""
        if not self.raw_schema:
            return "No schema data available"
        
        summary = self.get_table_summary()
        column_stats = self.get_column_statistics()
        
        report = []
        report.append("="*80)
        title = f"DATABASE SCHEMA EXTRACTION REPORT - {summary.get('report_type', 'ALL').upper()}"
        report.append(title)
        report.append("="*80)
        report.append("")
        
        # Database info
        db_info = self.raw_schema["database_info"]
        report.append(f"Database: {db_info['database_name']}")
        report.append(f"Host: {db_info['host']}:{db_info['port']}")
        report.append(f"Report Type: {summary.get('report_type', 'all')}")
        report.append(f"Extraction Date: {db_info['extraction_timestamp']}")
        
        # Filter info
        if self.raw_schema.get("table_filter"):
            report.append(f"Table Filter Applied: Yes ({len(self.raw_schema['table_filter'])} target tables)")
        else:
            report.append("Table Filter Applied: No (all tables)")
        
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"  Total Tables: {summary['total_tables']}")
        report.append(f"  Total Columns: {column_stats['total_columns']}")
        report.append(f"  Extraction Duration: {self.extraction_stats.get('extraction_duration', 0):.2f} seconds")
        
        if self.extraction_stats.get("failed_tables", 0) > 0:
            report.append(f"  Failed Tables: {self.extraction_stats['failed_tables']}")
        
        report.append("")
        
        # Data type distribution
        report.append("DATA TYPE DISTRIBUTION:")
        for data_type, count in sorted(column_stats['data_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / column_stats['total_columns']) * 100
            report.append(f"  {data_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Most common column names
        report.append("MOST COMMON COLUMN NAMES:")
        for name, count in column_stats['most_common_column_names'].items():
            report.append(f"  {name}: {count} occurrences")
        report.append("")
        
        # Table details
        report.append("TABLE DETAILS:")
        report.append("-" * 80)
        report.append(f"{'Table Name':<30} {'Columns':<8} {'Rows':<12} {'PKs':<4} {'FKs':<4} {'Indexes':<7}")
        report.append("-" * 80)
        
        for table in summary['tables']:
            row_count = f"{table['row_count']:,}" if table['row_count'] else "Unknown"
            report.append(f"{table['name']:<30} {table['columns']:<8} {row_count:<12} {table['primary_keys']:<4} {table['foreign_keys']:<4} {table['indexes']:<7}")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)

# Global schema extractor instance
schema_extractor = SchemaExtractor()