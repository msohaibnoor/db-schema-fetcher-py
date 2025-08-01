"""
Database connection management for MySQL Schema Fetcher
"""
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator, List
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import pandas as pd

from ..utils.config import config
from ..utils.logger import schema_logger, logger

class DatabaseConnector:
    """Handles MySQL database connections and operations with table filtering"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self._connection_retries = 3
        self._retry_delay = 2
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            schema_logger.log_info(f"Connecting to MySQL database: {config.database.host}:{config.database.port}/{config.database.database}")
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                config.database.connection_url,
                **config.database.connection_params,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            self._test_connection()
            
            # Initialize metadata
            self.metadata = MetaData()
            
            schema_logger.log_success("Database connection established successfully")
            return True
            
        except Exception as e:
            schema_logger.log_error(f"Failed to connect to database: {str(e)}")
            return False
    
    def _test_connection(self):
        """Test database connection with retries"""
        for attempt in range(self._connection_retries):
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                    return
            except OperationalError as e:
                if attempt < self._connection_retries - 1:
                    schema_logger.log_warning(f"Connection attempt {attempt + 1} failed, retrying in {self._retry_delay}s...")
                    time.sleep(self._retry_delay)
                else:
                    raise e
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Get database connection context manager"""
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def get_inspector(self):
        """Get SQLAlchemy inspector for schema introspection"""
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        return inspect(self.engine)
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQL query and return pandas DataFrame"""
        try:
            with self.get_connection() as conn:
                return pd.read_sql(query, conn, params=params)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_table_row_count(self, table_name: str) -> Optional[int]:
        """Get approximate row count for a table"""
        try:
            query = f"SELECT COUNT(*) as count FROM `{table_name}`"
            result = self.execute_query(query)
            return int(result.iloc[0]['count'])
        except Exception as e:
            logger.warning(f"Could not get row count for table {table_name}: {str(e)}")
            return None
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get sample data from a table"""
        try:
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.warning(f"Could not get sample data for table {table_name}: {str(e)}")
            return None
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table information"""
        inspector = self.get_inspector()
        
        info = {
            "name": table_name,
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "row_count": None,
            "sample_data": None
        }
        
        try:
            # Get column information
            columns = inspector.get_columns(table_name)
            for col in columns:
                column_info = {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default"),
                    "autoincrement": col.get("autoincrement", False),
                    "comment": col.get("comment", "")
                }
                info["columns"].append(column_info)
            
            # Get primary keys
            try:
                pk_constraint = inspector.get_pk_constraint(table_name)
                info["primary_keys"] = pk_constraint.get("constrained_columns", [])
            except Exception as e:
                logger.debug(f"Could not get primary keys for {table_name}: {str(e)}")
            
            # Get foreign keys
            if config.processing.include_foreign_keys:
                try:
                    foreign_keys = inspector.get_foreign_keys(table_name)
                    for fk in foreign_keys:
                        fk_info = {
                            "name": fk.get("name", ""),
                            "constrained_columns": fk.get("constrained_columns", []),
                            "referred_table": fk.get("referred_table", ""),
                            "referred_columns": fk.get("referred_columns", [])
                        }
                        info["foreign_keys"].append(fk_info)
                except Exception as e:
                    logger.debug(f"Could not get foreign keys for {table_name}: {str(e)}")
            
            # Get indexes
            if config.processing.include_indexes:
                try:
                    indexes = inspector.get_indexes(table_name)
                    for idx in indexes:
                        idx_info = {
                            "name": idx.get("name", ""),
                            "columns": idx.get("column_names", []),
                            "unique": idx.get("unique", False)
                        }
                        info["indexes"].append(idx_info)
                except Exception as e:
                    logger.debug(f"Could not get indexes for {table_name}: {str(e)}")
            
            # Get row count
            info["row_count"] = self.get_table_row_count(table_name)
            
            # Get sample data
            if config.processing.include_sample_data:
                sample_df = self.get_sample_data(table_name, config.processing.sample_rows_limit)
                if sample_df is not None and not sample_df.empty:
                    # Convert to JSON serializable format
                    info["sample_data"] = sample_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting info for table {table_name}: {str(e)}")
            raise
        
        return info
    
    def get_database_info(self, table_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get general database information with optional table filtering"""
        info = {
            "database_name": config.database.database,
            "host": config.database.host,
            "port": config.database.port,
            "charset": config.database.charset,
            "extraction_timestamp": pd.Timestamp.now().isoformat(),
            "total_tables": 0,
            "tables": [],
            "table_filter_applied": bool(table_filter)
        }
        
        try:
            inspector = self.get_inspector()
            all_table_names = inspector.get_table_names()
            
            # Apply table filter if provided
            if table_filter:
                # Find existing tables that match the filter
                filtered_tables = []
                missing_tables = []
                
                for table in table_filter:
                    if table in all_table_names:
                        filtered_tables.append(table)
                    else:
                        missing_tables.append(table)
                
                info["tables"] = filtered_tables
                info["total_tables"] = len(filtered_tables)
                info["filtered_from_total"] = len(all_table_names)
                info["missing_tables"] = missing_tables
                
                if missing_tables:
                    schema_logger.log_warning(f"Tables not found in database: {', '.join(missing_tables)}")
                
                schema_logger.log_info(f"Table filter applied: {len(filtered_tables)}/{len(all_table_names)} tables selected")
                
            else:
                info["tables"] = all_table_names
                info["total_tables"] = len(all_table_names)
            
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            raise
        
        return info
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            inspector = self.get_inspector()
            return table_name in inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False
    
    def validate_table_filter(self, table_filter: List[str]) -> Dict[str, List[str]]:
        """Validate that tables in filter exist in database"""
        try:
            inspector = self.get_inspector()
            all_tables = inspector.get_table_names()
            
            existing_tables = [table for table in table_filter if table in all_tables]
            missing_tables = [table for table in table_filter if table not in all_tables]
            
            return {
                "existing": existing_tables,
                "missing": missing_tables,
                "total_in_db": len(all_tables)
            }
        except Exception as e:
            logger.error(f"Error validating table filter: {str(e)}")
            return {"existing": [], "missing": table_filter, "total_in_db": 0}
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            schema_logger.log_info("Database connection closed")

# Global connector instance
db_connector = DatabaseConnector()