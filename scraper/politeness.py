"""robots.txt compliance and a fixed inter-request delay (PROJECT_PLAN.md
Phase 4 "Politeness"). Real network calls, so not unit-tested against live
sites; scraper/scrape.py is the only caller.
"""
from __future__ import annotations

import logging
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

USER_AGENT = "ThirdUmpireBot/0.1 (student project)"
MIN_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 3.0

logger = logging.getLogger(__name__)


def is_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    """True if robots.txt for `url`'s host allows fetching it. Fails open
    (True) if robots.txt can't be fetched or parsed, matching common scraper
    convention -- the explicit User-Agent and the rate limit below still
    apply regardless."""
    parsed = urlparse(url)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except OSError as exc:
        logger.warning("Couldn't fetch %s (%s); proceeding as allowed.", robots_url, exc)
        return True
    return parser.can_fetch(user_agent, url)


def polite_delay(seconds: float = MIN_DELAY_SECONDS) -> None:
    time.sleep(seconds)
