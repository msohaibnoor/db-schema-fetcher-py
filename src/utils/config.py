"""
Configuration management for MySQL Schema Fetcher
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Handle pydantic imports gracefully
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator
        PYDANTIC_V2 = False
        field_validator = validator
    except ImportError:
        # Fallback without pydantic
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        def Field(default=None, **kwargs):
            return default
        
        def field_validator(field_name):
            def decorator(func):
                return func
            return decorator
        
        PYDANTIC_V2 = False

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database connection configuration"""
    
    host: str = Field(default="localhost")
    port: int = Field(default=3306)
    database: str = Field()
    username: str = Field()
    password: str = Field()
    charset: str = Field(default="utf8mb4")
    
    # Connection pool settings
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    
    if PYDANTIC_V2:
        model_config = {
            'env_prefix': 'DB_',
            'env_file': '.env',
            'case_sensitive': False,
            'extra': 'ignore'
        }
    else:
        class Config:
            env_prefix = 'DB_'
            env_file = '.env'
            case_sensitive = False
            extra = 'ignore'
    
    def __init__(self, **kwargs):
        # Manual environment variable loading for more control
        env_data = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'database': os.getenv('DB_NAME'),
            'username': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'charset': os.getenv('DB_CHARSET', 'utf8mb4'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        }
        
        # Remove None values and update with any provided kwargs
        env_data = {k: v for k, v in env_data.items() if v is not None}
        env_data.update(kwargs)
        
        super().__init__(**env_data)
    
    @property
    def connection_url(self) -> str:
        """Generate SQLAlchemy connection URL"""
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?charset={self.charset}"
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for SQLAlchemy engine"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_pre_ping": True,  # Verify connections before use
            "pool_recycle": 3600,   # Recycle connections after 1 hour
        }

class ProcessingConfig(BaseSettings):
    """Processing and output configuration"""
    
    output_dir: Path = Field(default=Path("./output"))
    log_level: str = Field(default="INFO")
    
    # Schema extraction options
    include_sample_data: bool = Field(default=True)
    sample_rows_limit: int = Field(default=5)
    include_foreign_keys: bool = Field(default=True)
    include_indexes: bool = Field(default=True)
    
    # JSON output options
    pretty_json: bool = Field(default=True)
    json_indent: int = Field(default=2)
    
    def __init__(self, **kwargs):
        # Manual environment variable loading
        def str_to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val)
        
        env_data = {
            'output_dir': Path(os.getenv('OUTPUT_DIR', './output')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'include_sample_data': str_to_bool(os.getenv('INCLUDE_SAMPLE_DATA', 'true')),
            'sample_rows_limit': int(os.getenv('SAMPLE_ROWS_LIMIT', '5')),
            'include_foreign_keys': str_to_bool(os.getenv('INCLUDE_FOREIGN_KEYS', 'true')),
            'include_indexes': str_to_bool(os.getenv('INCLUDE_INDEXES', 'true')),
            'pretty_json': True,
            'json_indent': 2,
        }
        
        env_data.update(kwargs)
        super().__init__(**env_data)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Validate required environment variables first
        self._validate_required_env_vars()
        
        # Initialize configurations
        self.database = DatabaseConfig()
        self.processing = ProcessingConfig()
    
    def _validate_required_env_vars(self):
        """Validate required environment variables"""
        required_vars = {
            'DB_NAME': 'Database name',
            'DB_USER': 'Database username', 
            'DB_PASSWORD': 'Database password'
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            print("Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\nPlease create a .env file with the following variables:")
            print("DB_HOST=localhost")
            print("DB_PORT=3306")
            print("DB_NAME=your_database_name")
            print("DB_USER=your_username")
            print("DB_PASSWORD=your_password")
            print("DB_CHARSET=utf8mb4")
            raise ValueError(f"Missing required environment variables: {', '.join([v.split(' ')[0] for v in missing_vars])}")
    
    def get_output_path(self, filename: str) -> Path:
        """Get full output path for a file"""
        return self.processing.output_dir / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                # Don't include password in output
            },
            "processing": {
                "output_dir": str(self.processing.output_dir),
                "include_sample_data": self.processing.include_sample_data,
                "sample_rows_limit": self.processing.sample_rows_limit,
                "include_foreign_keys": self.processing.include_foreign_keys,
                "include_indexes": self.processing.include_indexes,
            }
        }

# Create a function to get config instead of global instance
def get_config() -> Config:
    """Get configuration instance"""
    return Config()

# For backward compatibility, create config only when accessed
_config = None

def __getattr__(name):
    global _config
    if name == 'config':
        if _config is None:
            _config = Config()
        return _config
    raise AttributeError(f"module has no attribute '{name}'")