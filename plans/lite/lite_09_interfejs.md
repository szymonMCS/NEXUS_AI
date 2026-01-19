## 9. INTERFEJS CLI + GRADIO

### 9.1 Główny Entry Point (CLI)

```python
#!/usr/bin/env python3
# nexus.py - Główny entry point NEXUS AI Lite

import asyncio
import argparse
from datetime import datetime, date
from typing import Optional

from data.collectors.fixture_collector import FixtureCollector
from data.collectors.data_enricher import DataEnricher
from evaluator.web_data_evaluator import WebDataEvaluator, filter_by_quality
from prediction.tennis_model import TennisModel
from prediction.basketball_model import BasketballModel
from prediction.value_calculator import ValueCalculator
from ranking.match_ranker import MatchRanker
from reports.report_generator import ReportGenerator


async def run_analysis(
    sport: str,
    target_date: str,
    min_quality: float = 45.0,
    top_n: int = 5,
    verbose: bool = True
) -> str:
    """
    Przeprowadza pełną analizę i generuje raport.

    Args:
        sport: "tennis" lub "basketball"
        target_date: Data w formacie YYYY-MM-DD
        min_quality: Minimalny próg jakości danych
        top_n: Ile betów w raporcie
        verbose: Czy wyświetlać progress

    Returns:
        Ścieżka do wygenerowanego raportu
    """
    if verbose:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 NEXUS AI Lite - Analiza On-Demand                        ║
╠══════════════════════════════════════════════════════════════╣
║  Sport: {sport.upper():<10}  Data: {target_date:<15}              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # 1. COLLECT FIXTURES
    if verbose:
        print("📅 [1/5] Zbieranie meczów z internetu...")

    collector = FixtureCollector()
    fixtures = await collector.collect_fixtures(sport, target_date)

    if not fixtures:
        print("❌ Nie znaleziono meczów na ten dzień!")
        return None

    if verbose:
        print(f"   ✅ Znaleziono {len(fixtures)} meczów\n")

    # 2. ENRICH DATA
    if verbose:
        print("🔍 [2/5] Wzbogacanie danych (newsy, statystyki, kursy)...")

    enricher = DataEnricher()
    enriched = await enricher.enrich_all(fixtures, sport, max_concurrent=3)

    if verbose:
        print(f"   ✅ Wzbogacono {len(enriched)} meczów\n")

    # 3. EVALUATE DATA QUALITY
    if verbose:
        print("📊 [3/5] Ewaluacja jakości danych z internetu...")

    evaluator = WebDataEvaluator()
    matches_with_quality = []

    for match in enriched:
        report = evaluator.evaluate(match, sport)
        matches_with_quality.append((match, report))

        if verbose and report.overall_score < min_quality:
            print(f"   ⚠️ {match['home']} vs {match['away']}: "
                  f"quality {report.overall_score:.0f}% (SKIP)")

    # Filter by quality
    quality_matches = filter_by_quality(matches_with_quality, min_quality)

    if verbose:
        print(f"   ✅ {len(quality_matches)}/{len(matches_with_quality)} "
              f"meczów przeszło filtr jakości (>= {min_quality}%)\n")

    if not quality_matches:
        print("❌ Żaden mecz nie przeszedł filtra jakości!")
        generator = ReportGenerator()
        content = generator._generate_no_bets_report(sport, target_date)
        return generator.save_report(content, sport, target_date)

    # 4. PREDICTIONS & VALUE
    if verbose:
        print("🧠 [4/5] Obliczanie predykcji i szukanie value...")

    model = TennisModel() if sport == "tennis" else BasketballModel()
    value_calc = ValueCalculator()

    matches_with_predictions = []

    for match, quality_report in quality_matches:
        # Predict
        prediction = model.predict(match)
        match["prediction"] = prediction.__dict__

        # Calculate value
        value_bet = value_calc.calculate_value(
            home_prob=prediction.home_win_prob,
            away_prob=prediction.away_win_prob,
            odds=match.get("odds", {}),
            league_type="medium",  # TODO: Klasyfikacja ligi
            quality_score=quality_report.overall_score
        )

        matches_with_predictions.append((match, quality_report, value_bet))

        if verbose and value_bet and value_bet.has_value:
            print(f"   💰 {match['home']} vs {match['away']}: "
                  f"edge +{value_bet.edge:.1%}")

    if verbose:
        value_count = sum(1 for _, _, v in matches_with_predictions if v and v.has_value)
        print(f"   ✅ Znaleziono {value_count} value betów\n")

    # 5. RANK & GENERATE REPORT
    if verbose:
        print("📝 [5/5] Generowanie raportu...")

    ranker = MatchRanker()
    top_bets = ranker.rank_bets(matches_with_predictions, top_n=top_n)

    generator = ReportGenerator()
    content = generator.generate_markdown(top_bets, sport, target_date)
    filepath = generator.save_report(content, sport, target_date)

    if verbose:
        print(f"   ✅ Raport zapisany: {filepath}\n")
        print("=" * 60)
        print(content)
        print("=" * 60)

    return filepath


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NEXUS AI Lite - System predykcji sportowych on-demand"
    )

    parser.add_argument(
        "--sport", "-s",
        choices=["tennis", "basketball"],
        default="tennis",
        help="Sport do analizy (default: tennis)"
    )

    parser.add_argument(
        "--date", "-d",
        default=str(date.today()),
        help="Data do analizy YYYY-MM-DD (default: dziś)"
    )

    parser.add_argument(
        "--min-quality", "-q",
        type=float,
        default=45.0,
        help="Minimalny próg jakości danych 0-100 (default: 45)"
    )

    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Liczba betów w raporcie (default: 5)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Tryb cichy - tylko output raportu"
    )

    args = parser.parse_args()

    # Run async
    filepath = asyncio.run(run_analysis(
        sport=args.sport,
        target_date=args.date,
        min_quality=args.min_quality,
        top_n=args.top,
        verbose=not args.quiet
    ))

    if filepath:
        print(f"\n✅ Gotowe! Raport: {filepath}")
    else:
        print("\n❌ Nie udało się wygenerować raportu")


if __name__ == "__main__":
    main()
```

### 9.2 Gradio App (Opcjonalnie)

```python
# ui/gradio_app.py

import gradio as gr
import asyncio
from datetime import date

# Import main analysis function
import sys
sys.path.append("..")
from nexus import run_analysis

def run_sync(sport: str, target_date: str, min_quality: float, top_n: int):
    """Wrapper synchroniczny dla Gradio"""
    filepath = asyncio.run(run_analysis(
        sport=sport,
        target_date=target_date,
        min_quality=min_quality,
        top_n=int(top_n),
        verbose=False
    ))

    if filepath:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return "❌ Nie udało się wygenerować raportu"


def create_app():
    """Tworzy aplikację Gradio"""

    with gr.Blocks(title="NEXUS AI Lite", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎯 NEXUS AI Lite
        ### System predykcji sportowych on-demand

        Wygeneruj raport z najlepszymi betami na dany dzień.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                sport = gr.Dropdown(
                    choices=["tennis", "basketball"],
                    value="tennis",
                    label="🏅 Sport"
                )

                target_date = gr.Textbox(
                    value=str(date.today()),
                    label="📅 Data (YYYY-MM-DD)"
                )

                min_quality = gr.Slider(
                    minimum=30,
                    maximum=80,
                    value=45,
                    step=5,
                    label="📊 Min. jakość danych (%)"
                )

                top_n = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="🏆 Liczba betów w raporcie"
                )

                btn = gr.Button("🚀 Generuj Raport", variant="primary")

            with gr.Column(scale=2):
                output = gr.Markdown(label="Raport")

        btn.click(
            fn=run_sync,
            inputs=[sport, target_date, min_quality, top_n],
            outputs=output
        )

        gr.Markdown("""
        ---
        **Uwagi:**
        - System zbiera dane z internetu (Sofascore, Flashscore, newsy)
        - Jakość danych jest weryfikowana przed analizą
        - Minimalna jakość 45% jest zalecana dla wiarygodnych predykcji
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
```
