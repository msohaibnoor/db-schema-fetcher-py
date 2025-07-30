"""
Schema mapping for natural language to database field mapping
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

from ..utils.config import config
from ..utils.logger import schema_logger, logger


class SchemaMapper:
    """Creates mappings between natural language terms and database schema elements"""

    def __init__(self):
        self.processed_schema: Dict[str, Any] = {}
        self.mappings: Dict[str, Any] = {}
        self.synonyms: Dict[str, List[str]] = {}
        self.field_mappings: Dict[str, Dict[str, Any]] = {}

    def load_processed_schema(self, schema_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load processed schema from JSON file"""
        if schema_path is None:
            schema_path = config.get_output_path("processed_schema.json")

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.processed_schema = json.load(f)

            schema_logger.log_success(
                f"Processed schema loaded from: {schema_path}")
            return self.processed_schema

        except Exception as e:
            schema_logger.log_error(
                f"Failed to load processed schema: {str(e)}")
            raise

    def create_mappings(self, processed_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive mappings for natural language queries"""
        schema_logger.log_section("CREATING SCHEMA MAPPINGS")

        if processed_schema:
            self.processed_schema = processed_schema

        if not self.processed_schema:
            raise ValueError(
                "No processed schema data available. Load schema first.")

        start_time = datetime.now()

        try:
            # Initialize mappings structure
            self.mappings = {
                "metadata": {
                    "creation_date": start_time.isoformat(),
                    "database_name": self.processed_schema.get("metadata", {}).get("database_name"),
                    "total_tables": len(self.processed_schema.get("tables", {})),
                    "mapping_version": "1.0.0"
                },
                "table_mappings": self._create_table_mappings(),
                "column_mappings": self._create_column_mappings(),
                "value_mappings": self._create_value_mappings(),
                "business_term_mappings": self._create_business_term_mappings(),
                "query_patterns": self._create_query_patterns(),
                "fuzzy_matching": self._create_fuzzy_matching_data()
            }

            # Calculate mapping statistics
            end_time = datetime.now()
            mapping_duration = (end_time - start_time).total_seconds()

            self.mappings["metadata"]["mapping_stats"] = {
                "creation_duration_seconds": round(mapping_duration, 2),
                "total_table_mappings": len(self.mappings["table_mappings"]),
                "total_column_mappings": sum(len(table_cols) for table_cols in self.mappings["column_mappings"].values()),
                "total_business_terms": len(self.mappings["business_term_mappings"]),
                "total_query_patterns": len(self.mappings["query_patterns"])
            }

            schema_logger.log_success(
                f"Schema mappings created in {mapping_duration:.2f} seconds")
            return self.mappings

        except Exception as e:
            schema_logger.log_error(f"Mapping creation failed: {str(e)}")
            raise

    def _create_table_mappings(self) -> Dict[str, Any]:
        """Create mappings for table names to natural language terms"""
        tables = self.processed_schema.get("tables", {})
        table_mappings = {}

        for table_name, table_info in tables.items():
            # Base mapping
            mapping = {
                "actual_name": table_name,
                "aliases": [table_name.lower()],
                "business_terms": [],
                "description": table_info.get("description", ""),
                "purpose": table_info.get("business_context", {}).get("likely_purpose", "unknown"),
                "domain": table_info.get("business_context", {}).get("domain", "general")
            }

            # Generate aliases based on table name
            aliases = self._generate_table_aliases(table_name)
            mapping["aliases"].extend(aliases)

            # Generate business terms based on context
            business_terms = self._generate_business_terms(
                table_name, table_info)
            mapping["business_terms"] = business_terms

            table_mappings[table_name] = mapping

        return table_mappings

    def _generate_table_aliases(self, table_name: str) -> List[str]:
        """Generate alternative names/aliases for a table"""
        aliases = []

        # Remove common prefixes/suffixes
        clean_name = table_name.lower()
        for prefix in ['tbl_', 'tb_', 't_']:
            if clean_name.startswith(prefix):
                aliases.append(clean_name[len(prefix):])

        # Split by underscores and create variations
        if '_' in table_name:
            parts = table_name.lower().split('_')

            # Individual parts
            aliases.extend(parts)

            # Different combinations
            if len(parts) >= 2:
                # First + last part
                aliases.append(f"{parts[0]} {parts[-1]}")
                # All parts joined with spaces
                aliases.append(' '.join(parts))

        # Pluralization handling
        singular_forms = self._get_singular_forms(table_name.lower())
        aliases.extend(singular_forms)

        plural_forms = self._get_plural_forms(table_name.lower())
        aliases.extend(plural_forms)

        # Remove duplicates and original name
        aliases = list(set(aliases))
        if table_name.lower() in aliases:
            aliases.remove(table_name.lower())

        return aliases

    def _get_singular_forms(self, word: str) -> List[str]:
        """Get singular forms of potentially plural words"""
        singular_forms = []

        # Simple pluralization rules
        if word.endswith('s') and len(word) > 3:
            # Remove 's'
            singular_forms.append(word[:-1])

        if word.endswith('ies'):
            # Replace 'ies' with 'y'
            singular_forms.append(word[:-3] + 'y')

        if word.endswith('es') and len(word) > 3:
            # Remove 'es'
            singular_forms.append(word[:-2])

        return singular_forms

    def _get_plural_forms(self, word: str) -> List[str]:
        """Get plural forms of potentially singular words"""
        plural_forms = []

        # Simple pluralization rules
        if not word.endswith('s'):
            plural_forms.append(word + 's')

        if word.endswith('y') and len(word) > 2:
            # Replace 'y' with 'ies'
            plural_forms.append(word[:-1] + 'ies')

        if word.endswith(('ch', 'sh', 's', 'x', 'z')):
            plural_forms.append(word + 'es')

        return plural_forms

    def _generate_business_terms(self, table_name: str, table_info: Dict[str, Any]) -> List[str]:
        """Generate business terminology for a table"""
        business_terms = []

        # Based on business context
        context = table_info.get("business_context", {})
        purpose = context.get("likely_purpose", "")
        domain = context.get("domain", "")

        # Map purposes to business terms
        purpose_mappings = {
            "user_management": ["users", "customers", "accounts", "members", "people"],
            "transaction_processing": ["orders", "purchases", "transactions", "sales", "payments"],
            "catalog_management": ["products", "items", "inventory", "catalog", "merchandise"],
            "audit_logging": ["logs", "history", "audit trail", "events", "activities"],
            "system_config": ["settings", "configuration", "parameters", "options"]
        }

        if purpose in purpose_mappings:
            business_terms.extend(purpose_mappings[purpose])

        # Based on column names (infer purpose from columns)
        columns = table_info.get("columns", [])
        column_names = [col["name"].lower() for col in columns]

        # If has email/phone columns, likely contact info
        if any('email' in col or 'phone' in col for col in column_names):
            business_terms.extend(
                ["contacts", "contact information", "directory"])

        # If has price/amount columns, likely financial
        if any(term in col for col in column_names for term in ['price', 'amount', 'cost', 'total']):
            business_terms.extend(
                ["financial data", "monetary information", "pricing"])

        # If has address columns, likely location data
        if any(term in col for col in column_names for term in ['address', 'city', 'state', 'zip']):
            business_terms.extend(
                ["addresses", "locations", "geographic data"])

        return list(set(business_terms))  # Remove duplicates

    def _create_column_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Create mappings for column names to natural language terms"""
        tables = self.processed_schema.get("tables", {})
        column_mappings = {}

        for table_name, table_info in tables.items():
            table_columns = {}

            for column in table_info.get("columns", []):
                col_name = column["name"]

                mapping = {
                    "actual_name": col_name,
                    "table": table_name,
                    "type": column.get("type", "unknown"),
                    "aliases": self._generate_column_aliases(col_name),
                    "business_meaning": column.get("business_meaning", "general_data"),
                    "description": column.get("description", ""),
                    "sample_values": self._get_sample_values_for_column(table_name, col_name),
                    "common_phrases": self._generate_column_phrases(col_name, column)
                }

                table_columns[col_name] = mapping

            column_mappings[table_name] = table_columns

        return column_mappings

    def _generate_column_aliases(self, column_name: str) -> List[str]:
        """Generate aliases for column names"""
        aliases = []

        # Remove common prefixes/suffixes
        clean_name = column_name.lower()

        # Remove table prefixes (e.g., user_id -> id)
        if '_' in clean_name:
            parts = clean_name.split('_')
            if len(parts) > 1:
                # Last part (e.g., user_id -> id)
                aliases.append(parts[-1])
                # All parts with spaces
                aliases.append(' '.join(parts))

        # Common column name variations
        name_variations = {
            'id': ['identifier', 'key', 'number', 'code'],
            'name': ['title', 'label', 'description'],
            'email': ['email address', 'e-mail', 'electronic mail'],
            'phone': ['telephone', 'phone number', 'contact number'],
            'created_at': ['creation date', 'date created', 'created on'],
            'updated_at': ['modification date', 'last modified', 'updated on'],
            'status': ['state', 'condition', 'flag'],
            'active': ['enabled', 'status', 'live'],
            'price': ['cost', 'amount', 'value'],
            'quantity': ['qty', 'count', 'number', 'amount']
        }

        for pattern, variations in name_variations.items():
            if pattern in clean_name:
                aliases.extend(variations)

        return list(set(aliases))

    def _get_sample_values_for_column(self, table_name: str, column_name: str) -> List[Any]:
        """Get sample values for a column from processed schema"""
        tables = self.processed_schema.get("tables", {})

        if table_name in tables:
            sample_values = tables[table_name].get("sample_values", {})
            return sample_values.get(column_name, [])

        return []

    def _generate_column_phrases(self, column_name: str, column_info: Dict[str, Any]) -> List[str]:
        """Generate natural language phrases that might refer to this column"""
        phrases = []

        col_name = column_name.lower()
        col_type = column_info.get("type", "unknown")
        business_meaning = column_info.get("business_meaning", "")

        # Generate phrases based on column name and type
        if col_name.endswith('_id') or col_name == 'id':
            entity_name = col_name.replace(
                '_id', '') if col_name != 'id' else 'record'
            phrases.extend([
                f"{entity_name} identifier",
                f"{entity_name} number",
                f"{entity_name} key",
                f"ID of {entity_name}"
            ])

        elif 'name' in col_name:
            phrases.extend([
                "name",
                "title",
                "label",
                "what is called"
            ])

        elif 'date' in col_name or 'time' in col_name:
            if 'created' in col_name:
                phrases.extend([
                    "when created",
                    "creation date",
                    "date created",
                    "created on"
                ])
            elif 'updated' in col_name or 'modified' in col_name:
                phrases.extend([
                    "when updated",
                    "last modified",
                    "modification date",
                    "updated on"
                ])
            else:
                phrases.extend([
                    "date",
                    "time",
                    "when",
                    "timestamp"
                ])

        elif 'status' in col_name or 'state' in col_name:
            phrases.extend([
                "status",
                "current state",
                "condition",
                "what status"
            ])

        elif 'count' in col_name or 'quantity' in col_name:
            phrases.extend([
                "how many",
                "count",
                "number of",
                "quantity"
            ])

        elif 'price' in col_name or 'amount' in col_name or 'cost' in col_name:
            phrases.extend([
                "how much",
                "price",
                "cost",
                "amount",
                "value"
            ])

        # Add generic phrases based on data type
        if col_type == 'string':
            phrases.extend([
                f"what {col_name}",
                f"{col_name} value",
                f"text of {col_name}"
            ])
        elif col_type in ['integer', 'decimal']:
            phrases.extend([
                f"number of {col_name}",
                f"{col_name} count",
                f"how much {col_name}"
            ])
        elif col_type == 'datetime':
            phrases.extend([
                f"when {col_name}",
                f"{col_name} date",
                f"time of {col_name}"
            ])

        return list(set(phrases))  # Remove duplicates

    def _create_value_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Create mappings for common values to their meanings"""
        tables = self.processed_schema.get("tables", {})
        value_mappings = {}

        for table_name, table_info in tables.items():
            sample_values = table_info.get("sample_values", {})

            for column_name, values in sample_values.items():
                if not values:
                    continue

                column_key = f"{table_name}.{column_name}"
                value_mappings[column_key] = {
                    "table": table_name,
                    "column": column_name,
                    "sample_values": values,
                    "value_patterns": self._analyze_value_patterns(values),
                    "suggested_filters": self._suggest_filter_values(column_name, values)
                }

        return value_mappings

    def _analyze_value_patterns(self, values: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in sample values"""
        if not values:
            return {}

        patterns = {
            "data_type": "mixed",
            "has_nulls": None in values,
            "unique_values": len(set(str(v) for v in values if v is not None)),
            "common_patterns": []
        }

        # Determine primary data type
        non_null_values = [v for v in values if v is not None]
        if non_null_values:
            if all(isinstance(v, (int, float)) for v in non_null_values):
                patterns["data_type"] = "numeric"
            elif all(isinstance(v, str) for v in non_null_values):
                patterns["data_type"] = "string"
            elif all(isinstance(v, bool) for v in non_null_values):
                patterns["data_type"] = "boolean"

        # Find common patterns for strings
        if patterns["data_type"] == "string":
            string_values = [str(v) for v in non_null_values]

            # Check for common formats
            if all(re.fullmatch(r'^[A-Z]{2,}$', v) for v in string_values):
                patterns["common_patterns"].append("uppercase_codes")
            elif all(re.fullmatch(r'^\d+$', v) for v in string_values):
                patterns["common_patterns"].append("numeric_strings")
            elif all('@' in v for v in string_values):
                patterns["common_patterns"].append("email_addresses")
            elif all(re.match(r'^\d{4}-\d{2}-\d{2}', v) for v in string_values):
                patterns["common_patterns"].append("dates")

        return patterns

    def _suggest_filter_values(self, column_name: str, values: List[Any]) -> List[str]:
        """Suggest natural language phrases for filtering by these values"""
        if not values:
            return []

        suggestions = []
        col_name_lower = column_name.lower()

        # Generate suggestions based on column name and values
        if 'status' in col_name_lower:
            for value in values[:3]:  # Top 3 values
                if value:
                    suggestions.append(f"where status is {value}")
                    suggestions.append(f"with {value} status")

        elif 'type' in col_name_lower:
            for value in values[:3]:
                if value:
                    suggestions.append(f"of type {value}")
                    suggestions.append(f"where type equals {value}")

        elif 'name' in col_name_lower:
            for value in values[:2]:
                if value:
                    suggestions.append(f"named {value}")
                    suggestions.append(f"with name {value}")

        else:
            # Generic suggestions
            for value in values[:2]:
                if value:
                    suggestions.append(f"where {column_name} = {value}")
                    suggestions.append(f"with {column_name} of {value}")

        return suggestions

    def _create_business_term_mappings(self) -> Dict[str, Any]:
        """Create mappings for business terms to database elements"""
        business_mappings = {}

        # Common business terms and their database equivalents
        common_terms = {
            "customers": {"tables": ["users", "customers", "clients"], "meaning": "customer_data"},
            "orders": {"tables": ["orders", "purchases", "transactions"], "meaning": "transaction_data"},
            "products": {"tables": ["products", "items", "inventory"], "meaning": "catalog_data"},
            "sales": {"tables": ["orders", "sales", "transactions"], "meaning": "sales_data"},
            "revenue": {"columns": ["amount", "total", "price", "revenue"], "meaning": "financial_amount"},
            "profit": {"columns": ["profit", "margin", "net_amount"], "meaning": "profit_data"},
            "inventory": {"tables": ["inventory", "stock", "products"], "meaning": "inventory_data"},
            "employees": {"tables": ["employees", "staff", "personnel"], "meaning": "employee_data"},
            "reports": {"tables": ["reports", "analytics", "summaries"], "meaning": "reporting_data"}
        }

        # Add table-specific business terms from processed schema
        tables = self.processed_schema.get("tables", {})
        for table_name, table_info in tables.items():
            business_terms = table_info.get("business_terms", [])
            for term in business_terms:
                if term not in business_mappings:
                    business_mappings[term] = {
                        "tables": [table_name],
                        "columns": [],
                        "meaning": table_info.get("business_context", {}).get("likely_purpose", "general_data")
                    }
                else:
                    if table_name not in business_mappings[term]["tables"]:
                        business_mappings[term]["tables"].append(table_name)

        # Merge with common terms
        for term, mapping in common_terms.items():
            if term in business_mappings:
                # Merge existing mapping
                business_mappings[term]["tables"].extend(
                    mapping.get("tables", []))
                business_mappings[term]["columns"].extend(
                    mapping.get("columns", []))
            else:
                business_mappings[term] = mapping

        return business_mappings

    def _create_query_patterns(self) -> List[Dict[str, Any]]:
        """Create common query patterns with their mappings"""
        patterns = [
            {
                "pattern": "show me all {table}",
                "sql_template": "SELECT * FROM {table}",
                "description": "Retrieve all records from a table",
                "parameters": ["table"]
            },
            {
                "pattern": "how many {table} are there",
                "sql_template": "SELECT COUNT(*) FROM {table}",
                "description": "Count total records in a table",
                "parameters": ["table"]
            },
            {
                "pattern": "show me {table} where {column} is {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} = '{value}'",
                "description": "Filter records by specific column value",
                "parameters": ["table", "column", "value"]
            },
            {
                "pattern": "show me recent {table}",
                "sql_template": "SELECT * FROM {table} ORDER BY {date_column} DESC LIMIT 10",
                "description": "Show most recent records",
                "parameters": ["table", "date_column"]
            },
            {
                "pattern": "find {table} created in {date_range}",
                "sql_template": "SELECT * FROM {table} WHERE {date_column} >= '{start_date}' AND {date_column} <= '{end_date}'",
                "description": "Filter records by date range",
                "parameters": ["table", "date_column", "start_date", "end_date"]
            },
            {
                "pattern": "show me top {number} {table} by {column}",
                "sql_template": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT {number}",
                "description": "Show top N records ordered by column",
                "parameters": ["table", "column", "number"]
            }
        ]

        return patterns

    def _create_fuzzy_matching_data(self) -> Dict[str, Any]:
        """Create data structures for fuzzy matching of terms"""
        fuzzy_data = {
            "all_table_names": [],
            "all_column_names": [],
            "all_business_terms": [],
            "similarity_threshold": 0.6
        }

        # Collect all terms for fuzzy matching
        tables = self.processed_schema.get("tables", {})

        for table_name, table_info in tables.items():
            fuzzy_data["all_table_names"].append(table_name)

            # Add table aliases
            table_mappings = self.mappings.get("table_mappings", {})
            if table_name in table_mappings:
                fuzzy_data["all_table_names"].extend(
                    table_mappings[table_name].get("aliases", []))
                fuzzy_data["all_business_terms"].extend(
                    table_mappings[table_name].get("business_terms", []))

            # Add column names
            for column in table_info.get("columns", []):
                col_name = column["name"]
                fuzzy_data["all_column_names"].append(
                    f"{table_name}.{col_name}")

                # Add column aliases
                column_mappings = self.mappings.get("column_mappings", {})
                if table_name in column_mappings and col_name in column_mappings[table_name]:
                    aliases = column_mappings[table_name][col_name].get(
                        "aliases", [])
                    fuzzy_data["all_column_names"].extend(
                        [f"{table_name}.{alias}" for alias in aliases])

        # Remove duplicates
        fuzzy_data["all_table_names"] = list(
            set(fuzzy_data["all_table_names"]))
        fuzzy_data["all_column_names"] = list(
            set(fuzzy_data["all_column_names"]))
        fuzzy_data["all_business_terms"] = list(
            set(fuzzy_data["all_business_terms"]))

        return fuzzy_data

    def find_similar_terms(self, query_term: str, term_type: str = "all") -> List[Dict[str, Any]]:
        """Find similar terms using fuzzy matching"""
        if not self.mappings:
            return []

        fuzzy_data = self.mappings.get("fuzzy_matching", {})
        threshold = fuzzy_data.get("similarity_threshold", 0.6)

        candidates = []

        if term_type in ["all", "table"]:
            candidates.extend([(term, "table")
                              for term in fuzzy_data.get("all_table_names", [])])

        if term_type in ["all", "column"]:
            candidates.extend([(term, "column")
                              for term in fuzzy_data.get("all_column_names", [])])

        if term_type in ["all", "business"]:
            candidates.extend([(term, "business_term")
                              for term in fuzzy_data.get("all_business_terms", [])])

        # Calculate similarities
        matches = []
        for candidate_term, candidate_type in candidates:
            similarity = SequenceMatcher(
                None, query_term.lower(), candidate_term.lower()).ratio()

            if similarity >= threshold:
                matches.append({
                    "term": candidate_term,
                    "type": candidate_type,
                    "similarity": similarity,
                    "original_query": query_term
                })

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return matches[:10]  # Return top 10 matches

    def save_mappings(self, output_path: Optional[Path] = None) -> Path:
        """Save schema mappings to JSON file"""
        if not self.mappings:
            raise ValueError(
                "No mapping data to save. Run create_mappings() first.")

        if output_path is None:
            output_path = config.get_output_path("schema_mappings.json")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.mappings,
                    f,
                    indent=config.processing.json_indent if config.processing.pretty_json else None,
                    ensure_ascii=False,
                    default=str
                )

            file_size = output_path.stat().st_size
            schema_logger.log_success(
                f"Schema mappings saved to: {output_path} ({file_size:,} bytes)")
            return output_path

        except Exception as e:
            schema_logger.log_error(
                f"Failed to save schema mappings: {str(e)}")
            raise


    def generate_mapping_summary(self) -> str:
        """Generate a summary report of created mappings"""
        if not self.mappings:
            return "No mapping data available"

        stats = self.mappings.get("metadata", {}).get("mapping_stats", {})

        summary = []
        summary.append("=" * 80)
        summary.append("SCHEMA MAPPING SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        # Overall statistics
        summary.append("MAPPING STATISTICS:")
        summary.append(f"  Total Tables: {stats.get('total_table_mappings', 0)}")
        summary.append(f"  Total Column Mappings: {stats.get('total_column_mappings', 0)}")
        summary.append(f"  Business Terms: {stats.get('total_business_terms', 0)}")
        summary.append(f"  Query Patterns: {stats.get('total_query_patterns', 0)}")
        summary.append(f"  Creation Time: {stats.get('creation_duration_seconds', 0):.2f} seconds")
        summary.append("")

        # Table mappings summary
        table_mappings = self.mappings.get("table_mappings", {})
        summary.append("TABLE MAPPINGS SAMPLE:")
        summary.append("-" * 40)

        for i, (table_name, mapping) in enumerate(list(table_mappings.items())[:5]):
            aliases = mapping.get("aliases", [])[:3]
            business_terms = mapping.get("business_terms", [])[:2]

            summary.append(f"Table: {table_name}")
            if aliases:
                summary.append(f"  Aliases: {', '.join(aliases)}")
            if business_terms:
                summary.append(f"  Business Terms: {', '.join(business_terms)}")
            summary.append(f"  Purpose: {mapping.get('purpose', 'unknown')}")
            summary.append("")

        if len(table_mappings) > 5:
            summary.append(f"... and {len(table_mappings) - 5} more tables")
            summary.append("")

        # Business term mappings summary
        business_mappings = self.mappings.get("business_term_mappings", {})
        summary.append("BUSINESS TERM MAPPINGS SAMPLE:")
        summary.append("-" * 40)

        for i, (term, mapping) in enumerate(list(business_mappings.items())[:5]):
            tables = mapping.get("tables", [])[:3]
            summary.append(f"Term: '{term}'")
            if tables:
                summary.append(f"  Tables: {', '.join(tables)}")
            summary.append(f"  Meaning: {mapping.get('meaning', 'unknown')}")
            summary.append("")

        if len(business_mappings) > 5:
            summary.append(f"... and {len(business_mappings) - 5} more business terms")
            summary.append("")

        # Query patterns summary
        query_patterns = self.mappings.get("query_patterns", [])
        if query_patterns:
            summary.append("QUERY PATTERNS SAMPLE:")
            summary.append("-" * 40)

            for i, pattern in enumerate(query_patterns[:3]):
                summary.append(f"Pattern: {pattern.get('pattern', '')}")
                summary.append(f"  Description: {pattern.get('description', '')}")
                summary.append(f"  SQL Template: {pattern.get('sql_template', '')}")
                summary.append("")

            if len(query_patterns) > 3:
                summary.append(f"... and {len(query_patterns) - 3} more query patterns")
                summary.append("")

        # Fuzzy matching info
        fuzzy_data = self.mappings.get("fuzzy_matching", {})
        if fuzzy_data:
            summary.append("FUZZY MATCHING CONFIGURATION:")
            summary.append("-" * 40)
            summary.append(f"Similarity Threshold: {fuzzy_data.get('similarity_threshold', 0.6)}")
            summary.append(f"Total Table Names: {len(fuzzy_data.get('all_table_names', []))}")
            summary.append(f"Total Column Names: {len(fuzzy_data.get('all_column_names', []))}")
            summary.append(f"Total Business Terms: {len(fuzzy_data.get('all_business_terms', []))}")
            summary.append("")

        summary.append("=" * 80)
        return "\n".join(summary)


# Global schema mapper instance
schema_mapper = SchemaMapper()

