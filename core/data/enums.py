from enum import Enum


class Sport(Enum):
    """Wspierane dyscypliny sportowe w NEXUS ML."""
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    FOOTBALL = "football"
    HANDBALL = "handball"
    TABLE_TENNIS = "table_tennis"
    GREYHOUND = "greyhound"

    def __str__(self) -> str:
        return self.value