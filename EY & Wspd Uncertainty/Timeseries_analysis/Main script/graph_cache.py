"""

Supporting script for Timeseries analysis.py
Whole job is to make pwetty graphs.

Graph Caching & Viewer System
------------------------------
Allows me to:
1. Cache graphs as you generate them
2. Browse cached graphs in PyCharm without re-running analysis
3. Only regenerate if source data changed
4. Export to various formats

Usage:
    from graph_cache import GraphCache

    cache = GraphCache(base_dir="Output_newnew")

    # Save a graph
    cache.save(fig, location="2023PA032", graph_type="scatter_predicted",
               metadata={"mae": 0.45, "r2": 0.95})

    # View all graphs for a location
    cache.view(location="2023PA032")

    # Export all for publication
    cache.export_publication_ready(location="2023PA032", output_dir="Publication")
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Dict, List, Optional, Tuple
import numpy as np

class GraphCache:
    """
    Manages caching, viewing, and exporting of analysis graphs.
    """

    def __init__(self, base_dir: str = "Output_newnew"):
        """
        Initialize graph cache system.

        Args:
            base_dir: Root directory for all outputs
        """
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "_graph_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Define graph categories and their display order
        self.graph_categories = {
            "overview": ["dashboard", "device_location"],
            "validation": ["scatter_predicted", "scatter_self", "residuals"],
            "temporal": ["monthly_corr", "availability_corr", "seasonal_pattern"],
            "error_analysis": ["mae_normalized_pred", "mae_normalized_self",
                              "mae_absolute_pred", "mae_absolute_self"],
            "comparative": ["trix_vs_deviation", "ranking"]
        }

        # Modern color scheme
        self.colors = {
            "primary": "#2E86AB",      # Professional blue
            "secondary": "#A23B72",    # Accent purple
            "success": "#06A77D",      # Green for good metrics
            "warning": "#F18F01",      # Orange for caution
            "danger": "#C73E1D",       # Red for errors
            "neutral": "#6C757D",      # Gray for reference
            "background": "#F8F9FA",   # Light background
            "text": "#212529"          # Dark text
        }

    def _get_data_hash(self, location: str) -> str:
        """
        Create hash of source data files to detect changes.

        Args:
            location: Project location ID

        Returns:
            MD5 hash of relevant data files
        """
        hash_md5 = hashlib.md5()

        # Find all relevant data files for this location
        location_dir = self.base_dir / location
        if not location_dir.exists():
            return "no_data"

        # Hash all .txt files in raw data
        data_files = []
        for root, _, files in os.walk(location_dir):
            if "raw data" in root.lower():
                data_files.extend([os.path.join(root, f) for f in files if f.endswith('.txt')])

        # Sort for consistent hashing
        data_files.sort()

        for filepath in data_files:
            try:
                with open(filepath, 'rb') as f:
                    # Hash first and last 1KB to detect changes without reading entire file
                    hash_md5.update(f.read(1024))
                    f.seek(-1024, 2)  # Seek to 1KB before end
                    hash_md5.update(f.read(1024))
            except Exception:
                continue

        return hash_md5.hexdigest()

    def save(self,
             fig: plt.Figure,
             location: str,
             graph_type: str,
             metadata: Optional[Dict] = None,
             dpi: int = 300) -> Path:
        """
        Save a graph to cache with metadata.

        Args:
            fig: Matplotlib figure object
            location: Project location ID (e.g., "2023PA032")
            graph_type: Type of graph (e.g., "scatter_predicted")
            metadata: Optional dict with graph-specific data
            dpi: Resolution for saved image

        Returns:
            Path to saved graph
        """
        # Create location-specific cache directory
        loc_cache = self.cache_dir / location
        loc_cache.mkdir(exist_ok=True, parents=True)
        loc_cache.mkdir(exist_ok=True, parents=True)

        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = loc_cache / f"{graph_type}.png"
        pickle_path = loc_cache / f"{graph_type}.pkl"
        meta_path = loc_cache / f"{graph_type}_meta.json"

        # Save high-res PNG
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        # Save pickle for later modification
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)

        # Save metadata
        meta = {
            "location": location,
            "graph_type": graph_type,
            "created": timestamp,
            "data_hash": self._get_data_hash(location),
            "dpi": dpi,
            **(metadata or {})
        }

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"✅ Cached: {location}/{graph_type}")

        plt.close(fig)
        return png_path

    def needs_update(self, location: str, graph_type: str) -> bool:
        """
        Check if cached graph needs regeneration.

        Args:
            location: Project location ID
            graph_type: Type of graph

        Returns:
            True if graph should be regenerated
        """
        meta_path = self.cache_dir / location / f"{graph_type}_meta.json"

        if not meta_path.exists():
            return True

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Check if data has changed
            current_hash = self._get_data_hash(location)
            return meta.get("data_hash") != current_hash

        except Exception:
            return True

    def view(self,
             location: Optional[str] = None,
             graph_type: Optional[str] = None,
             category: Optional[str] = None) -> None:
        """
        Display cached graphs in a grid layout.

        Args:
            location: Specific location to view (None = all)
            graph_type: Specific graph type to view (None = all)
            category: Graph category to view (None = all)
        """
        # Find matching graphs
        graphs = self._find_graphs(location, graph_type, category)

        if not graphs:
            print(f"❌ No cached graphs found")
            return

        n_graphs = len(graphs)

        if n_graphs == 1:
            # Single graph: full size
            self._display_single(graphs[0])
        else:
            # Multiple graphs: grid
            self._display_grid(graphs)

    def _find_graphs(self,
                     location: Optional[str],
                     graph_type: Optional[str],
                     category: Optional[str]) -> List[Tuple[Path, Dict]]:
        """Find graphs matching criteria."""
        matches = []

        # Determine which locations to search
        if location:
            search_dirs = [self.cache_dir / location]
        else:
            search_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]

        for loc_dir in search_dirs:
            if not loc_dir.exists():
                continue

            # Find all PNG files
            for png_path in loc_dir.glob("*.png"):
                meta_path = png_path.with_name(png_path.stem + "_meta.json")

                if not meta_path.exists():
                    continue

                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                    # Apply filters
                    if graph_type and meta["graph_type"] != graph_type:
                        continue

                    if category:
                        cat_graphs = self.graph_categories.get(category, [])
                        if meta["graph_type"] not in cat_graphs:
                            continue

                    matches.append((png_path, meta))

                except Exception as e:
                    print(f"⚠️ Error reading {meta_path}: {e}")
                    continue

        # Sort by location, then category, then graph type
        def sort_key(item):
            _, meta = item
            loc = meta["location"]
            gtype = meta["graph_type"]

            # Find category
            cat_order = 999
            for i, (cat, gtypes) in enumerate(self.graph_categories.items()):
                if gtype in gtypes:
                    cat_order = i
                    break

            return (loc, cat_order, gtype)

        matches.sort(key=sort_key)

        return matches

    def _display_single(self, graph_info: Tuple[Path, Dict]) -> None:
        """Display a single graph."""
        png_path, meta = graph_info

        fig, ax = plt.subplots(figsize=(12, 8))
        img = mpimg.imread(str(png_path))
        ax.imshow(img)
        ax.axis('off')

        title = f"{meta['location']} - {meta['graph_type'].replace('_', ' ').title()}"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def _display_grid(self, graphs: List[Tuple[Path, Dict]]) -> None:
        """Display multiple graphs in a grid."""
        n = len(graphs)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(6*cols, 4*rows))

        for i, (png_path, meta) in enumerate(graphs, 1):
            ax = fig.add_subplot(rows, cols, i)
            img = mpimg.imread(str(png_path))
            ax.imshow(img)
            ax.axis('off')

            title = f"{meta['location']}\n{meta['graph_type'].replace('_', ' ')}"
            ax.set_title(title, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def export_publication_ready(self,
                                 location: str,
                                 output_dir: str = "Publication",
                                 formats: List[str] = ['png', 'pdf', 'svg']) -> None:
        """
        Export all graphs for a location in publication-ready formats.

        Args:
            location: Project location ID
            output_dir: Output directory name
            formats: List of formats to export
        """
        export_path = self.base_dir / output_dir / location
        export_path.mkdir(exist_ok=True, parents=True)

        # Load all pickled figures
        loc_cache = self.cache_dir / location

        if not loc_cache.exists():
            print(f"❌ No cache found for {location}")
            return

        exported = 0

        for pkl_path in loc_cache.glob("*.pkl"):
            graph_type = pkl_path.stem

            try:
                # Load figure
                with open(pkl_path, 'rb') as f:
                    fig = pickle.load(f)

                # Export in each format
                for fmt in formats:
                    out_file = export_path / f"{graph_type}.{fmt}"
                    fig.savefig(out_file, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')

                exported += 1
                plt.close(fig)

            except Exception as e:
                print(f"⚠️ Error exporting {graph_type}: {e}")
                continue

        print(f"✅ Exported {exported} graphs to {export_path}")

    def list_locations(self) -> List[str]:
        """List all locations with cached graphs."""
        locations = [d.name for d in self.cache_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('_')]
        return sorted(locations)

    def summary(self) -> None:
        """Print summary of cached graphs."""
        locations = self.list_locations()

        print(f"\n{'='*60}")
        print(f"GRAPH CACHE SUMMARY")
        print(f"{'='*60}")
        print(f"Locations: {len(locations)}")
        print(f"Cache directory: {self.cache_dir}")

        for loc in locations:
            graphs = self._find_graphs(location=loc, graph_type=None, category=None)
            print(f"\n📍 {loc}: {len(graphs)} graphs")

            for cat, gtypes in self.graph_categories.items():
                cat_graphs = [g for g in graphs if g[1]["graph_type"] in gtypes]
                if cat_graphs:
                    print(f"  {cat}: {len(cat_graphs)}")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR YOUR MAIN SCRIPT
# =============================================================================

def setup_cache(base_dir: str = "Output_newnew") -> GraphCache:
    """Initialize cache for use in main script."""
    return GraphCache(base_dir)


# Example usage in PyCharm console:
if __name__ == "__main__":
    cache = GraphCache()

    # View all graphs
    cache.summary()

    # View specific location
    # cache.view(location="2023PA032")

    # View by category
    # cache.view(category="validation")

    # Export for publication
    # cache.export_publication_ready("2023PA032")