#!/usr/bin/env python3
"""Laboratorio 1: Minería de texto y procesamiento de lenguaje natural.

Analiza cuatro eBooks de ciencia ficción de Project Gutenberg:
- The Eyes Have It — Philip K. Dick — eBook 31516
- Keep Out — Fredric Brown — eBook 29142
- Navy Day — Harry Harrison — eBook 30019
- 2 B R 0 2 B — Kurt Vonnegut — eBook 21279

El script:
1) obtiene/prepara el corpus;
2) tokeniza y elimina stopwords;
3) calcula los 15 términos más frecuentes por obra;
4) genera una nube de palabras por obra;
5) calcula TF-IDF con TF relativa e IDF = ln(N/df);
6) grafica los 15 términos más característicos por obra;
7) compara frecuencia vs. TF-IDF;
8) calcula asociaciones por correlación de coocurrencia en segmentos;
9) exporta gráficos y tablas CSV.

Compatible con Python 3.10+ y Google Colab.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
LOCAL_CORPUS_DIR = ROOT / "local_corpus"  # utilizado solo si existe una copia local

for directory in (ASSETS_DIR, RESULTS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Book:
    gutenberg_id: int
    title: str
    author: str
    slug: str
    start_anchor: str
    end_anchor: str

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt"

    @property
    def local_snapshot(self) -> Path:
        return LOCAL_CORPUS_DIR / f"{self.gutenberg_id}_{self.slug}.txt"


BOOKS = [
    Book(
        31516, "The Eyes Have It", "Philip K. Dick", "the_eyes_have_it",
        "It was quite by accident I discovered this incredible invasion of Earth",
        "I have absolutely no stomach for it.",
    ),
    Book(
        29142, "Keep Out", "Fredric Brown", "keep_out",
        "Daptine is the secret of it.",
        "Keep off!",
    ),
    Book(
        30019, "Navy Day", "Harry Harrison", "navy_day",
        "General Wingrove looked at the rows of faces without seeing them.",
        "the world awaits your decision.",
    ),
    Book(
        21279, "2 B R 0 2 B", "Kurt Vonnegut", "2_b_r_0_2_b",
        "Everything was perfectly swell.",
        "the deepest thanks of all is from future generations.",
    ),
]

CUSTOM_STOPWORDS = {
    "said", "say", "says", "saying", "asked", "tell", "told",
    "project", "gutenberg", "ebook", "ebooks", "transcriber", "illustration",
    "chapter", "chapters", "copyright", "license", "www", "http", "https",
    "author", "story", "stories", "jr", "mr", "mrs", "miss",
    "like", "thing", "things", "person", "persons", "maybe", "book", "read", "reading",
    "came", "come", "comes", "went", "going", "look", "looked", "looking", "right", "left",
    "time", "day", "room", "think", "thought", "spoke", "really", "man", "men",
    "don't", "didn't", "wouldn't", "couldn't", "isn't", "aren't", "it's", "that's",
    "there's", "i'm", "you're", "we're", "they're", "wasn't", "won't", "can't",
    "could", "would", "should",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ---------------------------------------------------------------------------
# Corpus y preprocesamiento
# ---------------------------------------------------------------------------

def download_text(book: Book) -> str:
    """Descarga un eBook de Gutenberg o usa una instantánea local si existe."""
    if book.local_snapshot.exists():
        return book.local_snapshot.read_text(encoding="utf-8")

    cache = DATA_DIR / f"pg{book.gutenberg_id}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    response = requests.get(book.url, timeout=45)
    response.raise_for_status()
    response.encoding = "utf-8"
    cache.write_text(response.text, encoding="utf-8")
    return response.text


def strip_gutenberg_boilerplate(text: str) -> str:
    """Conserva solamente el contenido entre las marcas START/END."""
    start = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    if start and end and end.start() > start.end():
        return text[start.end():end.start()]
    return text


def extract_story_body(text: str, book: Book) -> str:
    """Aísla de forma reproducible el relato y excluye portada/notas editoriales.

    Las anclas son frases del primer y último párrafo del eBook oficial. Si una
    edición futura cambia esas frases, se conserva como respaldo el contenido
    completo situado entre las marcas START/END de Project Gutenberg.
    """
    body = strip_gutenberg_boilerplate(text)
    start = body.find(book.start_anchor)
    end = body.rfind(book.end_anchor)
    if start != -1 and end != -1 and end >= start:
        return body[start:end + len(book.end_anchor)]
    return body


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        word = raw.strip("'")
        if len(word) <= 2:
            continue
        if word in STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def build_corpus() -> tuple[dict[str, str], dict[str, list[str]]]:
    texts: dict[str, str] = {}
    tokens: dict[str, list[str]] = {}
    for book in BOOKS:
        raw = download_text(book)
        body = extract_story_body(raw, book)
        texts[book.title] = body
        tokens[book.title] = tokenize(body)
    return texts, tokens

# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------

def frequency_table(tokens_by_book: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for book, tokens in tokens_by_book.items():
        for word, count in Counter(tokens).items():
            rows.append((book, word, count))
    return pd.DataFrame(rows, columns=["libro", "termino", "frecuencia"])


def top_n_frequency(freq: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        freq.sort_values(["libro", "frecuencia", "termino"], ascending=[True, False, True])
        .groupby("libro", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def tfidf_table(freq: pd.DataFrame, tokens_by_book: dict[str, list[str]]) -> pd.DataFrame:
    n_docs = len(tokens_by_book)
    doc_freq = freq.groupby("termino")["libro"].nunique().to_dict()
    totals = {book: len(tokens) for book, tokens in tokens_by_book.items()}

    out = freq.copy()
    out["tf"] = out.apply(lambda r: r["frecuencia"] / totals[r["libro"]], axis=1)
    out["df"] = out["termino"].map(doc_freq)
    out["idf"] = out["df"].map(lambda df: math.log(n_docs / df))
    out["tf_idf"] = out["tf"] * out["idf"]
    return out.sort_values("tf_idf", ascending=False).reset_index(drop=True)


def top_n_tfidf(tfidf: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        tfidf.sort_values(["libro", "tf_idf", "frecuencia"], ascending=[True, False, False])
        .groupby("libro", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def compare_lists(top_freq: pd.DataFrame, top_tfidf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    overlap_rows = []
    for book in [b.title for b in BOOKS]:
        f = top_freq[top_freq.libro == book].sort_values("frecuencia", ascending=False).reset_index(drop=True)
        t = top_tfidf[top_tfidf.libro == book].sort_values("tf_idf", ascending=False).reset_index(drop=True)
        f_words = set(f.termino)
        t_words = set(t.termino)
        common = sorted(f_words & t_words)
        overlap_rows.append({
            "libro": book,
            "coincidencias": len(common),
            "porcentaje": 100 * len(common) / 15,
            "terminos_comunes": ", ".join(common),
        })
        for i in range(15):
            rows.append({
                "libro": book,
                "posicion": i + 1,
                "termino_frecuente": f.loc[i, "termino"],
                "frecuencia": int(f.loc[i, "frecuencia"]),
                "termino_caracteristico": t.loc[i, "termino"],
                "tf_idf": float(t.loc[i, "tf_idf"]),
            })
    return pd.DataFrame(rows), pd.DataFrame(overlap_rows)


def association_table(
    tokens: list[str],
    target_candidates: Iterable[str],
    segment_size: int = 80,
    min_word_frequency: int = 3,
    top_n: int = 12,
) -> tuple[str, pd.DataFrame]:
    """Asociación por correlación de presencia en segmentos contiguos.

    Cada segmento actúa como un pseudodocumento. Para cada palabra se crea un
    vector binario de presencia/ausencia y se calcula la correlación de Pearson
    con el vector del término objetivo. Se elige el primer candidato distintivo
    que aparezca en >=2 y <100% de los segmentos.
    """
    segments = [tokens[i:i + segment_size] for i in range(0, len(tokens), segment_size)]
    segment_sets = [set(s) for s in segments if s]
    global_counts = Counter(tokens)

    eligible = {
        w for w, c in global_counts.items()
        if c >= min_word_frequency and sum(w in s for s in segment_sets) >= 2
    }

    target = None
    target_vec = None
    for candidate in target_candidates:
        vec = np.array([1.0 if candidate in s else 0.0 for s in segment_sets])
        if candidate in eligible and 1 < vec.sum() < len(vec):
            target = candidate
            target_vec = vec
            break

    if target is None:
        # Respaldo: escoger la palabra elegible de mayor frecuencia con variación.
        for candidate, _ in global_counts.most_common():
            vec = np.array([1.0 if candidate in s else 0.0 for s in segment_sets])
            if candidate in eligible and 1 < vec.sum() < len(vec):
                target = candidate
                target_vec = vec
                break

    if target is None or target_vec is None:
        raise RuntimeError("No fue posible seleccionar un término objetivo con variación suficiente.")

    rows = []
    for word in eligible:
        if word == target:
            continue
        vec = np.array([1.0 if word in s else 0.0 for s in segment_sets])
        if vec.std() == 0 or target_vec.std() == 0:
            continue
        corr = float(np.corrcoef(target_vec, vec)[0, 1])
        if np.isfinite(corr):
            rows.append((word, corr, global_counts[word], int(vec.sum())))

    result = pd.DataFrame(rows, columns=["termino_asociado", "correlacion", "frecuencia", "segmentos"])
    result = result[result.correlacion > 0].sort_values(
        ["correlacion", "frecuencia"], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)
    return target, result

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def save_bar_terms(df: pd.DataFrame, value_col: str, title: str, xlabel: str, path: Path) -> None:
    data = df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.barh(data["termino"], data[value_col], color="#2C5F8A")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_wordcloud(counter: Counter, title: str, path: Path) -> None:
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=100,
        collocations=False,
        colormap="Blues",
        random_state=42,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(counter)
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overlap(overlap: pd.DataFrame, path: Path) -> None:
    data = overlap.sort_values("coincidencias", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    ax.barh(data["libro"], data["coincidencias"], color="#4CAF50")
    ax.set_xlim(0, 15)
    ax.set_xlabel("Términos coincidentes entre ambos top 15")
    ax.set_title("Coincidencia: frecuencia vs. TF-IDF", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.2)
    for y, value in enumerate(data["coincidencias"]):
        ax.text(value + 0.15, y, f"{int(value)}/15", va="center")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_associations(df: pd.DataFrame, target: str, book: str, path: Path) -> None:
    data = df.sort_values("correlacion", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.barh(data["termino_asociado"], data["correlacion"], color="#E67E22")
    ax.set_xlabel("Correlación de coocurrencia por segmentos")
    ax.set_title(f"Asociaciones de ‘{target}’ — {book}", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def main() -> None:
    _, tokens_by_book = build_corpus()

    summary_rows = []
    for book in BOOKS:
        toks = tokens_by_book[book.title]
        summary_rows.append({
            "gutenberg_id": book.gutenberg_id,
            "libro": book.title,
            "autor": book.author,
            "tokens_limpios": len(toks),
            "vocabulario": len(set(toks)),
            "url": f"https://www.gutenberg.org/ebooks/{book.gutenberg_id}",
        })
    corpus_summary = pd.DataFrame(summary_rows)

    freq = frequency_table(tokens_by_book)
    top_freq = top_n_frequency(freq)
    tfidf = tfidf_table(freq, tokens_by_book)
    top_tfidf = top_n_tfidf(tfidf)
    comparison, overlap = compare_lists(top_freq, top_tfidf)

    # Exportaciones tabulares.
    corpus_summary.to_csv(RESULTS_DIR / "00_resumen_corpus.csv", index=False)
    freq.to_csv(RESULTS_DIR / "01_frecuencias_completas.csv", index=False)
    top_freq.to_csv(RESULTS_DIR / "02_top15_frecuentes.csv", index=False)
    tfidf.to_csv(RESULTS_DIR / "03_tfidf_completo.csv", index=False)
    top_tfidf.to_csv(RESULTS_DIR / "04_top15_tfidf.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "05_comparacion_frecuencia_tfidf.csv", index=False)
    overlap.to_csv(RESULTS_DIR / "06_coincidencias.csv", index=False)

    # Gráficos de frecuencia y nubes por libro.
    graphic_manifest: dict[str, str] = {}
    for idx, book in enumerate(BOOKS, start=1):
        d = top_freq[top_freq.libro == book.title]
        path = ASSETS_DIR / f"{idx:02d}_frecuentes_{book.slug}.png"
        save_bar_terms(d, "frecuencia", f"15 términos más frecuentes — {book.title}", "Frecuencia absoluta", path)
        graphic_manifest[f"freq_{book.slug}"] = path.name

    for idx, book in enumerate(BOOKS, start=5):
        counter = Counter(tokens_by_book[book.title])
        path = ASSETS_DIR / f"{idx:02d}_nube_{book.slug}.png"
        save_wordcloud(counter, f"Nube de palabras — {book.title}", path)
        graphic_manifest[f"cloud_{book.slug}"] = path.name

    for idx, book in enumerate(BOOKS, start=9):
        d = top_tfidf[top_tfidf.libro == book.title]
        path = ASSETS_DIR / f"{idx:02d}_tfidf_{book.slug}.png"
        save_bar_terms(d, "tf_idf", f"15 términos más característicos — {book.title}", "TF-IDF", path)
        graphic_manifest[f"tfidf_{book.slug}"] = path.name

    overlap_path = ASSETS_DIR / "13_comparacion_solapamiento.png"
    save_overlap(overlap, overlap_path)
    graphic_manifest["overlap"] = overlap_path.name

    # Asociaciones: dos términos característicos de dos libros distintos.
    association_books = [BOOKS[1], BOOKS[2]]  # Keep Out y Navy Day
    association_info = []
    for idx, book in enumerate(association_books, start=14):
        candidates = (
            top_tfidf[top_tfidf.libro == book.title]
            .sort_values("tf_idf", ascending=False)["termino"]
            .tolist()
        )
        target, assoc = association_table(tokens_by_book[book.title], candidates)
        assoc.insert(0, "termino_objetivo", target)
        assoc.insert(0, "libro", book.title)
        assoc_path_csv = RESULTS_DIR / f"07_asociaciones_{book.slug}_{target}.csv"
        assoc.to_csv(assoc_path_csv, index=False)
        image_path = ASSETS_DIR / f"{idx:02d}_asociaciones_{book.slug}_{target}.png"
        save_associations(assoc, target, book.title, image_path)
        graphic_manifest[f"assoc_{book.slug}"] = image_path.name
        association_info.append({
            "libro": book.title,
            "termino_objetivo": target,
            "archivo_csv": assoc_path_csv.name,
            "archivo_grafica": image_path.name,
            "top_asociaciones": assoc.to_dict(orient="records"),
        })

    # Resumen legible por el informe Typst y para auditoría.
    result_summary = {
        "corpus": corpus_summary.to_dict(orient="records"),
        "top_frecuentes": {
            b.title: top_freq[top_freq.libro == b.title][["termino", "frecuencia"]]
            .sort_values("frecuencia", ascending=False).to_dict(orient="records")
            for b in BOOKS
        },
        "top_tfidf": {
            b.title: top_tfidf[top_tfidf.libro == b.title][["termino", "frecuencia", "tf", "idf", "tf_idf"]]
            .sort_values("tf_idf", ascending=False).to_dict(orient="records")
            for b in BOOKS
        },
        "overlap": overlap.to_dict(orient="records"),
        "asociaciones": association_info,
        "graficas": graphic_manifest,
    }
    (RESULTS_DIR / "resumen_resultados.json").write_text(
        json.dumps(result_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nResumen del corpus")
    print(corpus_summary.to_string(index=False))
    print("\nTérminos característicos principales")
    for book in BOOKS:
        d = top_tfidf[top_tfidf.libro == book.title].nlargest(5, "tf_idf")
        print(f"- {book.title}: " + ", ".join(d.termino))
    print("\nAsociaciones")
    for item in association_info:
        print(f"- {item['libro']}: término objetivo = {item['termino_objetivo']}")
    print(f"\nResultados guardados en: {RESULTS_DIR}")
    print(f"Gráficas guardadas en: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
