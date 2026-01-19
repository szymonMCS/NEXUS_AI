## 11. DEPLOYMENT I MONITORING

### 11.1 `docker-compose.yml`

```yaml
version: '3.8'

services:
  nexus-app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "7860:7860"
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ODDS_API_KEY=${ODDS_API_KEY}
      - API_TENNIS_KEY=${API_TENNIS_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://nexus:${DB_PASSWORD}@postgres:5432/nexus
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  betting-floor:
    build:
      context: .
      dockerfile: docker/Dockerfile
    command: python betting_floor.py
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ODDS_API_KEY=${ODDS_API_KEY}
      - API_TENNIS_KEY=${API_TENNIS_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://nexus:${DB_PASSWORD}@postgres:5432/nexus
      - REDIS_URL=redis://redis:6379/0
      - RUN_EVERY_N_MINUTES=30
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  mcp-news:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/news_server.py
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
    restart: unless-stopped

  mcp-evaluation:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/evaluation_server.py
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped

  mcp-odds:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    command: python mcp_servers/odds_server.py
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=nexus
      - POSTGRES_USER=nexus
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafanadata:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
  grafanadata:
```

### 11.2 Monthly Cost Estimate

| Service | Plan | Cost | Limit |
|---------|------|-------|-------|
| **Brave Search API** | Free | $0 | 2,000 req/month |
| **Serper API** | Free | $0 | 2,500 req/month |
| **NewsAPI** | Free | $0 | 100 req/day |
| **The Odds API** | Starter | $79 | 10,000 req/month |
| **api-tennis.com** | Basic | ~$30 | 5,000 req/month |
| **Anthropic Claude** | Pay-as-you-go | ~$50-100 | ~500k tokens/day |
| **Hosting (VPS)** | Basic | ~$20 | 4GB RAM |
| **Redis Cloud** | Free | $0 | 30MB |
| **PostgreSQL** | Self-hosted | $0 | Included in VPS |

**TOTAL: ~$180-230/month**

---

## PODSUMOWANIE

System **NEXUS AI v2.0** integruje najlepsze rozwiązania z obu specyfikacji:

### Kluczowe Komponenty:

1. **NewsAggregator** - agregacja z Brave, Serper, NewsAPI z deduplikacją i sortowaniem
2. **NewsValidator** - quality, freshness, reliability, diversity scores
3. **InjuryExtractor** - ekstrakcja kontuzji przez LLM
4. **DataEvaluator** - kompleksowa ocena jakości przed analizą (KLUCZOWY!)
5. **QualityScorer** - korekty predykcji na podstawie jakości danych
6. **MatchRanker** - ranking Top 3 po composite score
7. **MCP Servers** - modularne backendy dla każdej funkcjonalności
8. **Gradio UI** - interaktywny interfejs z Top 3 cards
9. **BettingFloor** - główna pętla z pełnym workflow

### Innowacje:

- **Quality-Aware Value** - edge jest mnożony przez quality multiplier
- **Adaptive Thresholds** - wyższe wymagania dla lig niepopularnych
- **Multi-Source News** - 3 źródła newsów + ekstrakcja kontuzji
- **Composite Scoring** - edge × quality × confidence
- **Risk Levels** - automatyczne określanie poziomu ryzyka

### Następne Kroki:

1. Implementacja `data/odds/pl_scraper.py` (Playwright scraper)
2. Implementacja `core/models/tennis_model.py` i `basketball_model.py`
3. Integracja z rzeczywistymi API
4. Testy backtestingowe
5. Deployment na VPS

---
