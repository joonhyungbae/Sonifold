"""
Capture paper figure: Screenshot of the web app (paper \\includegraphics{figures/web.png}).
Requires: dev server running (cd webapp && npm run dev), and: pip install playwright && playwright install chromium
Run from project root: python scripts/figures/capture_demo_screenshot.py
Output: figures/web.png (app viewport; for canvas-only use the in-app "Export for paper" button).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
out_path = root / "figures" / "web.png"


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install playwright: pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto("http://localhost:5173", wait_until="networkidle", timeout=15000)
        time.sleep(1)
        page.select_option("select", value="torus")
        time.sleep(2)
        page.screenshot(path=str(out_path))
        browser.close()
    print("Saved", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
