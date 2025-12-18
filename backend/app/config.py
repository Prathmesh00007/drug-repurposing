"""Configuration management for Route A backend."""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API
    api_title: str = "Route A: Drug Repurposing Agent System"
    api_version: str = "0.1.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # External APIs
    opentargets_gql_url: str = os.getenv(
        "OPENTARGETS_GQL_URL", 
        "https://api.platform.opentargets.org/api/v4/graphql"
    )
    chembl_base_url: str = os.getenv(
        "CHEMBL_BASE_URL", 
        "https://www.ebi.ac.uk/chembl/api/data"
    )
    clinicaltrials_base_url: str = os.getenv(
        "CLINICALTRIALS_BASE_URL",
        "https://www.clinicaltrials.gov/api/v2"
    )
    patentsview_base_url: str = os.getenv(
        "PATENTSVIEW_BASE_URL",
        "https://api.patentsview.org/patents/query"
    )
    comtrade_base_url: str = os.getenv(
        "COMTRADE_BASE_URL",
        "https://comtradeapi.un.org/public/api/get"
    )
    
    # Redis (optional)
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # DRKG Configuration 
    drkg_enabled: bool = os.getenv("DRKG_ENABLED", "true").lower() == "true"
    drkg_cache_dir: str = os.getenv("DRKG_CACHE_DIR", "./data/drkg")

    
    # Data
    data_dir: str = os.getenv("DATA_DIR", "/data/runs")
    max_cache_age_seconds: int = 86400  # 24h
    
    # Constraints
    max_runtime_seconds: int = 600
    max_external_calls_per_run: int = 100
    max_candidates_to_return: int = 5
    
    # Timeouts
    http_timeout_seconds: int = 30
    neo4j_timeout_seconds: int = 30

    # Keys
    gemini_api_key: str | None = None
    kaggle_api_token: str | None = None
    kaggle_model_slug: str | None = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
