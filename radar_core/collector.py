# Re-export collector from radar.collector for backward compatibility
from radar.collector import collect_sources


__all__ = ["collect_sources"]
