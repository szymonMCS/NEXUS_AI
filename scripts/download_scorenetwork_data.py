#!/usr/bin/env python3
"""
Skrypt do pobierania wszystkich plikow CSV ze strony SCORE Network.
Pobiera dane do folderu D:\ScoreNetworkData
"""

import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
from pathlib import Path

# Konfiguracja
BASE_URL = 'https://data.scorenetwork.org/'
OUTPUT_DIR = Path('D:/ScoreNetworkData')
DOWNLOAD_DELAY = 1  # sekundy miedzy pobraniami

# Tworzenie folderu
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_all_dataset_pages():
    """Pobierz liste wszystkich stron datasetow."""
    print("Zbieranie listy stron datasetow...")
    response = requests.get(BASE_URL, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    pages = []
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        if href and '.html' in href:
            # Pomin strony nawigacyjne
            if any(x in href for x in ['index', 'by-', 'submit', 'sources']):
                continue
            
            full_url = urljoin(BASE_URL, href)
            pages.append({
                'name': link.get_text(strip=True),
                'url': full_url,
                'sport': href.split('/')[0] if '/' in href else 'other'
            })
    
    print(f"Znaleziono {len(pages)} stron datasetow")
    return pages

def find_csv_links(page_url):
    """Znajdz wszystkie linki do CSV na danej stronie."""
    try:
        response = requests.get(page_url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        csv_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '.csv' in href.lower():
                full_url = urljoin(page_url, href)
                csv_links.append({
                    'filename': os.path.basename(href),
                    'url': full_url
                })
        
        return csv_links
    except Exception as e:
        print(f"  Blad przy {page_url}: {e}")
        return []

def download_csv(csv_info, output_path):
    """Pobierz plik CSV."""
    try:
        response = requests.get(csv_info['url'], timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return len(response.content)
    except Exception as e:
        print(f"    Blad pobierania {csv_info['filename']}: {e}")
        return 0

def main():
    print("=" * 60)
    print("POBIERANIE DANYCH Z SCORE NETWORK")
    print("=" * 60)
    print(f"Folder docelowy: {OUTPUT_DIR}")
    print()
    
    # Pobierz liste stron
    pages = get_all_dataset_pages()
    
    # Statystyki
    total_csvs = 0
    downloaded_csvs = 0
    total_size = 0
    
    # Przetworz kazda strone
    for i, page in enumerate(pages, 1):
        print(f"\n[{i}/{len(pages)}] {page['name']}")
        print(f"  URL: {page['url']}")
        
        # Znajdz linki do CSV
        csv_links = find_csv_links(page['url'])
        
        if not csv_links:
            print("  Brak plikow CSV")
            continue
        
        print(f"  Znaleziono {len(csv_links)} plikow CSV")
        total_csvs += len(csv_links)
        
        # Utworz folder dla sportu
        sport_dir = OUTPUT_DIR / page['sport']
        sport_dir.mkdir(exist_ok=True)
        
        # Pobierz kazdy CSV
        for csv_info in csv_links:
            output_path = sport_dir / csv_info['filename']
            
            # Sprawdz czy plik juz istnieje
            if output_path.exists():
                print(f"    -> {csv_info['filename']} (juz istnieje)")
                downloaded_csvs += 1
                total_size += output_path.stat().st_size
                continue
            
            print(f"    -> Pobieranie {csv_info['filename']}...", end=' ')
            size = download_csv(csv_info, output_path)
            
            if size > 0:
                print(f"OK ({size:,} bajtow)")
                downloaded_csvs += 1
                total_size += size
            
            time.sleep(DOWNLOAD_DELAY)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"Przetworzono stron: {len(pages)}")
    print(f"Znaleziono plikow CSV: {total_csvs}")
    print(f"Pobrano/Zaktualizowano: {downloaded_csvs}")
    print(f"Calkowity rozmiar: {total_size / (1024*1024):.2f} MB")
    print(f"\nDane zapisano w: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
