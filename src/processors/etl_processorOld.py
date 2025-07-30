"""
ETL Processing for database schema transformation and enrichment
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

from ..utils.config import config
from ..utils.logger import schema_logger, logger

class ETLProcessor:
    """Processes and transforms raw schema data for AI/LLM consumption"""
    
    def __init__(self):
        self.raw_schema: Dict[str, Any] = {}
        self.processed_schema: Dict[str, Any] = {}
        self.processing_stats: Dict[str, Any] = {}
    
    def load_raw_schema(self, schema_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load raw schema from JSON file"""
        if schema_path is None:
            schema_path = config.get_output_path("raw_schema.json")
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.raw_schema = json.load(f)
            
            schema_logger.log_success(f"Raw schema loaded from: {schema_path}")
            return self.raw_schema
            
        except Exception as e:
            schema_logger.log_error(f"Failed to load raw schema: {str(e)}")
            raise
    
    def process_schema(self, raw_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process raw schema into LLM-friendly format"""
        schema_logger.log_section("PROCESSING SCHEMA FOR AI/LLM")
        
        if raw_schema:
            self.raw_schema = raw_schema
        
        if not self.raw_schema:
            raise ValueError("No raw schema data available. Load schema first.")
        
        start_time = datetime.now()
        
        try:
            # Initialize processed schema structure
            self.processed_schema = {
                "metadata": self._create_processing_metadata(),
                "database_summary": self._create_database_summary(),
                "tables": self._process_tables(),
                "relationships": self._extract_relationships(),
                "data_patterns": self._analyze_data_patterns(),
                "llm_context": self._create_llm_context()
            }
            
            # Calculate processing stats
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            self.processing_stats = {
                "processing_start": start_time.isoformat(),
                "processing_end": end_time.isoformat(),
                "processing_duration_seconds": round(processing_duration, 2),
                "tables_processed": len(self.processed_schema["tables"]),
                "relationships_found": len(self.processed_schema["relationships"]),
                "total_columns_processed": sum(len(table["columns"]) for table in self.processed_schema["tables"].values())
            }
            
            self.processed_schema["metadata"]["processing_stats"] = self.processing_stats
            
            schema_logger.log_success(f"Schema processing completed in {processing_duration:.2f} seconds")
            return self.processed_schema
            
        except Exception as e:
            schema_logger.log_error(f"Schema processing failed: {str(e)}")
            raise
    
    def _create_processing_metadata(self) -> Dict[str, Any]:
        """Create metadata for processed schema"""
        original_metadata = self.raw_schema.get("extraction_metadata", {})
        
        return {
            "original_extraction_date": original_metadata.get("extraction_date"),
            "processing_date": datetime.now().isoformat(),
            "processor_version": "1.0.0",
            "database_name": self.raw_schema.get("database_info", {}).get("database_name"),
            "processing_config": config.to_dict()
        }
    
    def _create_database_summary(self) -> Dict[str, Any]:
        """Create high-level database summary"""
        db_info = self.raw_schema.get("database_info", {})
        tables = self.raw_schema.get("tables", {})
        
        total_columns = sum(len(table.get("columns", [])) for table in tables.values())
        total_rows = sum(table.get("row_count", 0) or 0 for table in tables.values())
        
        return {
            "database_name": db_info.get("database_name"),
            "total_tables": len(tables),
            "total_columns": total_columns,
            "estimated_total_rows": total_rows,
            "largest_tables": self._get_largest_tables(),
            "table_categories": self._categorize_tables()
        }
    
    def _get_largest_tables(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the largest tables by row count"""
        tables = self.raw_schema.get("tables", {})
        table_sizes = []
        
        for table_name, table_info in tables.items():
            row_count = table_info.get("row_count", 0) or 0
            table_sizes.append({
                "name": table_name,
                "row_count": row_count,
                "column_count": len(table_info.get("columns", []))
            })
        
        # Sort by row count and return top N
        table_sizes.sort(key=lambda x: x["row_count"], reverse=True)
        return table_sizes[:limit]
    
    def _categorize_tables(self) -> Dict[str, List[str]]:
        """Categorize tables based on naming patterns"""
        tables = self.raw_schema.get("tables", {})
        categories = {
            "user_data": [],
            "system_config": [],
            "logs_audit": [],
            "lookup_reference": [],
            "transaction": [],
            "reporting": [],
            "other": []
        }
        
        for table_name in tables.keys():
            table_lower = table_name.lower()
            
            # User/Customer data
            if any(keyword in table_lower for keyword in ['user', 'customer', 'client', 'person', 'profile']):
                categories["user_data"].append(table_name)
            # System configuration
            elif any(keyword in table_lower for keyword in ['config', 'setting', 'parameter', 'option']):
                categories["system_config"].append(table_name)
            # Logs and audit
            elif any(keyword in table_lower for keyword in ['log', 'audit', 'history', 'event', 'activity']):
                categories["logs_audit"].append(table_name)
            # Lookup/Reference tables
            elif any(keyword in table_lower for keyword in ['lookup', 'reference', 'type', 'category', 'status']):
                categories["lookup_reference"].append(table_name)
            # Transaction tables
            elif any(keyword in table_lower for keyword in ['order', 'transaction', 'payment', 'invoice', 'purchase']):
                categories["transaction"].append(table_name)
            # Reporting tables
            elif any(keyword in table_lower for keyword in ['report', 'summary', 'metric', 'stat', 'analytics']):
                categories["reporting"].append(table_name)
            else:
                categories["other"].append(table_name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _process_tables(self) -> Dict[str, Any]:
        """Process individual tables for LLM consumption"""
        tables = self.raw_schema.get("tables", {})
        processed_tables = {}
        
        for table_name, table_info in tables.items():
            processed_tables[table_name] = {
                "name": table_name,
                "description": self._generate_table_description(table_name, table_info),
                "columns": self._process_columns(table_info.get("columns", [])),
                "primary_key": table_info.get("primary_keys", []),
                "foreign_keys": self._process_foreign_keys(table_info.get("foreign_keys", [])),
                "indexes": self._process_indexes(table_info.get("indexes", [])),
                "row_count": table_info.get("row_count"),
                "sample_values": self._extract_sample_values(table_info.get("sample_data", [])),
                "business_context": self._infer_business_context(table_name, table_info),
                "common_queries": self._suggest_common_queries(table_name, table_info)
            }
        
        return processed_tables
    
    def _generate_table_description(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Generate human-readable table description"""
        columns = table_info.get("columns", [])
        row_count = table_info.get("row_count", 0)
        
        description_parts = []
        
        # Basic description
        description_parts.append(f"Table '{table_name}' contains {len(columns)} columns")
        if row_count:
            description_parts.append(f"with approximately {row_count:,} records")
        
        # Identify key columns
        key_columns = []
        for col in columns[:5]:  # Look at first 5 columns
            col_name = col["name"].lower()
            if "id" in col_name or "key" in col_name:
                key_columns.append(col["name"])
        
        if key_columns:
            description_parts.append(f"Key columns include: {', '.join(key_columns)}")
        
        return ". ".join(description_parts) + "."
    
    def _process_columns(self, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process column information for LLM understanding"""
        processed_columns = []
        
        for col in columns:
            processed_col = {
                "name": col["name"],
                "type": self._normalize_data_type(col["type"]),
                "nullable": col.get("nullable", True),
                "description": self._generate_column_description(col),
                "business_meaning": self._infer_column_meaning(col["name"]),
                "sample_values": None  # Will be populated from sample data
            }
            processed_columns.append(processed_col)
        
        return processed_columns
    
    def _normalize_data_type(self, sql_type: str) -> str:
        """Normalize SQL data types to common categories"""
        sql_type_lower = str(sql_type).lower()
        
        # String types
        if any(t in sql_type_lower for t in ['varchar', 'char', 'text', 'string']):
            return 'string'
        # Integer types
        elif any(t in sql_type_lower for t in ['int', 'integer', 'bigint', 'smallint', 'tinyint']):
            return 'integer'
        # Decimal/Float types
        elif any(t in sql_type_lower for t in ['decimal', 'numeric', 'float', 'double', 'real']):
            return 'decimal'
        # Date/Time types
        elif any(t in sql_type_lower for t in ['date', 'time', 'timestamp', 'datetime']):
            return 'datetime'
        # Boolean
        elif any(t in sql_type_lower for t in ['bool', 'boolean', 'bit']):
            return 'boolean'
        # JSON
        elif 'json' in sql_type_lower:
            return 'json'
        # Binary
        elif any(t in sql_type_lower for t in ['blob', 'binary', 'varbinary']):
            return 'binary'
        else:
            return 'other'
    
    def _generate_column_description(self, column: Dict[str, Any]) -> str:
        """Generate human-readable column description"""
        parts = []
        
        col_name = column["name"]
        col_type = column["type"]
        nullable = column.get("nullable", True)
        default = column.get("default")
        autoincrement = column.get("autoincrement", False)
        
        # Base description
        parts.append(f"Column '{col_name}' of type {col_type}")
        
        # Nullable
        if not nullable:
            parts.append("(required)")
        
        # Auto increment
        if autoincrement:
            parts.append("(auto-incrementing)")
        
        # Default value
        if default is not None:
            parts.append(f"with default value '{default}'")
        
        return " ".join(parts)
    
    def _infer_column_meaning(self, column_name: str) -> str:
        """Infer business meaning from column name"""
        name_lower = column_name.lower()
        
        # ID columns
        if name_lower.endswith('_id') or name_lower == 'id':
            return 'identifier'
        # Name columns
        elif 'name' in name_lower:
            return 'name_reference'
        # Date/Time columns
        elif any(keyword in name_lower for keyword in ['date', 'time', 'created', 'updated', 'modified']):
            return 'temporal'
        # Status/State columns
        elif any(keyword in name_lower for keyword in ['status', 'state', 'flag', 'active', 'enabled']):
            return 'status_indicator'
        # Count/Amount columns
        elif any(keyword in name_lower for keyword in ['count', 'amount', 'quantity', 'total', 'sum']):
            return 'numeric_measure'
        # Description/Comment columns
        elif any(keyword in name_lower for keyword in ['description', 'comment', 'note', 'remarks']):
            return 'descriptive_text'
        # Email/Contact columns
        elif any(keyword in name_lower for keyword in ['email', 'phone', 'contact']):
            return 'contact_information'
        else:
            return 'general_data'
    
    def _process_foreign_keys(self, foreign_keys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process foreign key relationships"""
        processed_fks = []
        
        for fk in foreign_keys:
            processed_fk = {
                "name": fk.get("name", ""),
                "columns": fk.get("constrained_columns", []),
                "references_table": fk.get("referred_table", ""),
                "references_columns": fk.get("referred_columns", []),
                "relationship_description": self._describe_relationship(fk)
            }
            processed_fks.append(processed_fk)
        
        return processed_fks
    
    def _describe_relationship(self, foreign_key: Dict[str, Any]) -> str:
        """Generate human-readable relationship description"""
        source_cols = foreign_key.get("constrained_columns", [])
        target_table = foreign_key.get("referred_table", "")
        target_cols = foreign_key.get("referred_columns", [])
        
        if source_cols and target_table and target_cols:
            return f"References {target_table}({', '.join(target_cols)}) via {', '.join(source_cols)}"
        return "Foreign key relationship"
    
    def _process_indexes(self, indexes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process index information"""
        processed_indexes = []
        
        for idx in indexes:
            processed_idx = {
                "name": idx.get("name", ""),
                "columns": idx.get("columns", []),
                "unique": idx.get("unique", False),
                "type": "unique" if idx.get("unique", False) else "standard"
            }
            processed_indexes.append(processed_idx)
        
        return processed_indexes
    
    def _extract_sample_values(self, sample_data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract sample values for each column"""
        if not sample_data:
            return {}
        
        sample_values = {}
        
        for row in sample_data:
            for column, value in row.items():
                if column not in sample_values:
                    sample_values[column] = []
                
                # Only keep unique, non-null values
                if value is not None and value not in sample_values[column]:
                    sample_values[column].append(value)
        
        # Limit sample values to prevent overly large outputs
        for column in sample_values:
            sample_values[column] = sample_values[column][:10]
        
        return sample_values
    
    def _infer_business_context(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Infer business context and purpose of the table"""
        columns = table_info.get("columns", [])
        column_names = [col["name"].lower() for col in columns]
        
        context = {
            "likely_purpose": "unknown",
            "domain": "general",
            "entity_type": "unknown",
            "confidence": 0.0
        }
        
        table_lower = table_name.lower()
        
        # Determine purpose based on naming patterns and columns
        if any(keyword in table_lower for keyword in ['user', 'customer', 'client']):
            context.update({
                "likely_purpose": "user_management",
                "domain": "user_administration",
                "entity_type": "user_entity",
                "confidence": 0.8
            })
        elif any(keyword in table_lower for keyword in ['order', 'purchase', 'transaction']):
            context.update({
                "likely_purpose": "transaction_processing",
                "domain": "e_commerce",
                "entity_type": "transaction_entity",
                "confidence": 0.9
            })
        elif any(keyword in table_lower for keyword in ['product', 'item', 'inventory']):
            context.update({
                "likely_purpose": "catalog_management",
                "domain": "inventory",
                "entity_type": "product_entity",
                "confidence": 0.8
            })
        elif any(keyword in table_lower for keyword in ['log', 'audit', 'history']):
            context.update({
                "likely_purpose": "audit_logging",
                "domain": "system_monitoring",
                "entity_type": "log_entity",
                "confidence": 0.9
            })
        
        return context
    
    def _suggest_common_queries(self, table_name: str, table_info: Dict[str, Any]) -> List[str]:
        """Suggest common query patterns for the table"""
        columns = table_info.get("columns", [])
        primary_keys = table_info.get("primary_keys", [])
        
        queries = []
        
        # Basic queries
        queries.append(f"SELECT * FROM {table_name}")
        queries.append(f"SELECT COUNT(*) FROM {table_name}")
        
        # Primary key queries
        if primary_keys:
            pk = primary_keys[0]
            queries.append(f"SELECT * FROM {table_name} WHERE {pk} = ?")
        
        # Date-based queries if date columns exist
        date_columns = [col["name"] for col in columns if "date" in col["name"].lower() or "time" in col["name"].lower()]
        if date_columns:
            date_col = date_columns[0]
            queries.append(f"SELECT * FROM {table_name} WHERE {date_col} >= ?")
            queries.append(f"SELECT * FROM {table_name} ORDER BY {date_col} DESC")
        
        # Status-based queries
        status_columns = [col["name"] for col in columns if any(keyword in col["name"].lower() for keyword in ['status', 'state', 'active'])]
        if status_columns:
            status_col = status_columns[0]
            queries.append(f"SELECT * FROM {table_name} WHERE {status_col} = ?")
        
        return queries[:5]  # Limit to 5 suggestions
    
    def _extract_relationships(self) -> List[Dict[str, Any]]:
        """Extract all relationships between tables"""
        relationships = []
        tables = self.raw_schema.get("tables", {})
        
        for table_name, table_info in tables.items():
            foreign_keys = table_info.get("foreign_keys", [])
            
            for fk in foreign_keys:
                relationship = {
                    "from_table": table_name,
                    "from_columns": fk.get("constrained_columns", []),
                    "to_table": fk.get("referred_table", ""),
                    "to_columns": fk.get("referred_columns", []),
                    "relationship_type": "one_to_many",  # Default assumption
                    "description": f"{table_name} references {fk.get('referred_table', '')}"
                }
                relationships.append(relationship)
        
        return relationships
    
    def _analyze_data_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the data for AI context"""
        tables = self.raw_schema.get("tables", {})
        
        patterns = {
            "naming_conventions": self._analyze_naming_conventions(tables),
            "common_columns": self._find_common_columns(tables),
            "data_type_distribution": self._analyze_data_types(tables),
            "table_size_distribution": self._analyze_table_sizes(tables)
        }
        
        return patterns
    
    def _analyze_naming_conventions(self, tables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze naming conventions used in the database"""
        conventions = {
            "table_naming": "unknown",
            "column_naming": "unknown",
            "uses_prefixes": False,
            "uses_underscores": False,
            "common_suffixes": []
        }
        
        table_names = list(tables.keys())
        all_columns = []
        
        for table_info in tables.values():
            all_columns.extend([col["name"] for col in table_info.get("columns", [])])
        
        # Analyze table naming
        if all('_' in name for name in table_names):
            conventions["table_naming"] = "snake_case"
            conventions["uses_underscores"] = True
        elif all(name.islower() for name in table_names):
            conventions["table_naming"] = "lowercase"
        
        # Analyze column naming
        if all('_' in name for name in all_columns):
            conventions["column_naming"] = "snake_case"
            conventions["uses_underscores"] = True
        
        # Find common suffixes
        suffixes = {}
        for col_name in all_columns:
            if '_' in col_name:
                suffix = col_name.split('_')[-1]
                suffixes[suffix] = suffixes.get(suffix, 0) + 1
        
        # Get most common suffixes
        common_suffixes = sorted(suffixes.items(), key=lambda x: x[1], reverse=True)[:5]
        conventions["common_suffixes"] = [suffix for suffix, count in common_suffixes if count > 1]
        
        return conventions
    
    def _find_common_columns(self, tables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find columns that appear in multiple tables"""
        column_counts = {}
        
        for table_name, table_info in tables.items():
            for col in table_info.get("columns", []):
                col_name = col["name"]
                if col_name not in column_counts:
                    column_counts[col_name] = {"count": 0, "tables": []}
                column_counts[col_name]["count"] += 1
                column_counts[col_name]["tables"].append(table_name)
        
        # Find columns appearing in multiple tables
        common_columns = []
        for col_name, info in column_counts.items():
            if info["count"] > 1:
                common_columns.append({
                    "column_name": col_name,
                    "appears_in_tables": info["count"],
                    "tables": info["tables"]
                })
        
        # Sort by frequency
        common_columns.sort(key=lambda x: x["appears_in_tables"], reverse=True)
        return common_columns[:10]  # Top 10 most common
    
    def _analyze_data_types(self, tables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution of data types"""
        type_counts = {}
        total_columns = 0
        
        for table_info in tables.values():
            for col in table_info.get("columns", []):
                normalized_type = self._normalize_data_type(col["type"])
                type_counts[normalized_type] = type_counts.get(normalized_type, 0) + 1
                total_columns += 1
        
        # Calculate percentages
        type_distribution = {}
        for data_type, count in type_counts.items():
            type_distribution[data_type] = {
                "count": count,
                "percentage": round((count / total_columns) * 100, 2)
            }
        
        return type_distribution
    
    def _analyze_table_sizes(self, tables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table size distribution"""
        sizes = []
        
        for table_info in tables.values():
            row_count = table_info.get("row_count", 0) or 0
            sizes.append(row_count)
        
        if not sizes:
            return {}
        
        return {
            "total_rows": sum(sizes),
            "average_rows_per_table": round(sum(sizes) / len(sizes), 2),
            "largest_table_rows": max(sizes),
            "smallest_table_rows": min(sizes),
            "tables_with_data": len([s for s in sizes if s > 0])
        }
    
    def _create_llm_context(self) -> Dict[str, Any]:
        """Create optimized context for LLM consumption"""
        tables = self.processed_schema.get("tables", {})
        
        # Create simplified table descriptions for LLM
        table_summaries = {}
        for table_name, table_info in tables.items():
            table_summaries[table_name] = {
                "description": table_info["description"],
                "key_columns": [col["name"] for col in table_info["columns"][:5]],  # Top 5 columns
                "primary_key": table_info["primary_key"],
                "row_count": table_info["row_count"],
                "business_purpose": table_info["business_context"]["likely_purpose"]
            }
        
        return {
            "database_description": f"Database '{self.processed_schema['metadata']['database_name']}' with {len(tables)} tables",
            "table_summaries": table_summaries,
            "common_relationships": self.processed_schema.get("relationships", [])[:10],  # Top 10 relationships
            "query_suggestions": self._generate_query_suggestions()
        }
    
    def _generate_query_suggestions(self) -> List[str]:
        """Generate example queries that could be asked"""
        suggestions = [
            "Show me all records from [table_name]",
            "How many records are in [table_name]?",
            "What are the column names in [table_name]?",
            "Show me recent records from [table_name]",
            "Find records in [table_name] where [column] equals [value]"
        ]
        
        return suggestions
    
    def save_processed_schema(self, output_path: Optional[Path] = None) -> Path:
        """Save processed schema to JSON file"""
        if not self.processed_schema:
            raise ValueError("No processed schema data to save. Run process_schema() first.")
        
        if output_path is None:
            output_path = config.get_output_path("processed_schema.json")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.processed_schema,
                    f,
                    indent=config.processing.json_indent if config.processing.pretty_json else None,
                    ensure_ascii=False,
                    default=str
                )
            
            file_size = output_path.stat().st_size
            schema_logger.log_success(f"Processed schema saved to: {output_path} ({file_size:,} bytes)")
            return output_path
            
        except Exception as e:
            schema_logger.log_error(f"Failed to save processed schema: {str(e)}")
            raise

# Global ETL processor instance
etl_processor = ETLProcessor()