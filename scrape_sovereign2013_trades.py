#!/usr/bin/env python3
"""Export complete Polymarket trade history for a handle or wallet.

Examples:
  python scrape_sovereign2013_trades.py
  python scrape_sovereign2013_trades.py --user 0xabc... --endpoint trades
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# Known profile mapping from publicly reported account metadata.
KNOWN_HANDLES: dict[str, str] = {
    "sovereign2013": "0xee613b3fc183ee44f9da9c05f53e2da107e3debf",
}


class FetchError(RuntimeError):
    """Raised when all retries are exhausted for HTTP requests."""


def fetch_json(url: str, timeout: int = 30, retries: int = 5, backoff: float = 1.4) -> Any:
    """Fetch JSON with retry/backoff for transient errors and throttling."""
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PolymarketTradeScraper/1.0)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            # Retry likely-transient statuses.
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        sleep_s = (backoff**attempt) + random.uniform(0.0, 0.35)
        time.sleep(sleep_s)

    raise FetchError(f"Request failed for {url}: {last_error}")


def resolve_profile_address(handle_or_address: str) -> str:
    """Resolve a Polymarket profile handle to its proxy wallet address."""
    if ADDRESS_RE.match(handle_or_address):
        return handle_or_address.lower()

    handle = handle_or_address.strip().lstrip("@").lower()
    if handle in KNOWN_HANDLES:
        return KNOWN_HANDLES[handle]

    # 1) Gamma public search
    public_search_url = f"{GAMMA_API_BASE}/public-search?" + urllib.parse.urlencode({"query": handle})
    try:
        payload = fetch_json(public_search_url)
        candidates = payload if isinstance(payload, list) else [payload]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_handle = str(item.get("handle") or item.get("name") or "").lower()
            candidate = str(item.get("proxyWallet") or item.get("wallet") or "").lower()
            if item_handle == handle and ADDRESS_RE.match(candidate):
                return candidate
    except Exception:
        pass

    # 2) Direct profile lookup variants
    endpoints = (
        f"{GAMMA_API_BASE}/profiles/{urllib.parse.quote(handle)}",
        f"{DATA_API_BASE}/profile/{urllib.parse.quote(handle)}",
    )
    for endpoint in endpoints:
        try:
            payload = fetch_json(endpoint)
        except Exception:
            continue
        if isinstance(payload, dict):
            for key in ("proxyWallet", "wallet", "address", "profileAddress"):
                candidate = str(payload.get(key, "")).lower()
                if ADDRESS_RE.match(candidate):
                    return candidate

    raise ValueError(
        f"Could not resolve '{handle_or_address}' to a wallet address. "
        "Pass --user with a 0x... wallet or add a mapping in KNOWN_HANDLES."
    )


def fetch_all_records(
    endpoint: str,
    user_address: str,
    limit: int,
    sleep_s: float,
    start: int | None,
    end: int | None,
) -> list[dict[str, Any]]:
    """Fetch all records from `trades` or `activity` endpoints with offset pagination."""
    if endpoint not in {"trades", "activity"}:
        raise ValueError("endpoint must be one of: trades, activity")

    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    offset = 0

    while True:
        params: dict[str, Any] = {
            "user": user_address,
            "limit": limit,
            "offset": offset,
            "sortDirection": "ASC",
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        if endpoint == "trades":
            params["takerOnly"] = "false"
        else:
            params["type"] = "TRADE"

        url = f"{DATA_API_BASE}/{endpoint}?{urllib.parse.urlencode(params, doseq=True)}"
        page = fetch_json(url)
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected response type at offset={offset}: {type(page)!r}")

        if not page:
            break

        for row in page:
            if not isinstance(row, dict):
                continue
            key = (
                row.get("transactionHash"),
                row.get("asset"),
                row.get("price"),
                row.get("size") or row.get("amount"),
                row.get("timestamp"),
                row.get("side"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)

        if len(page) < limit:
            break

        offset += limit
        time.sleep(sleep_s)

    out.sort(key=lambda x: (x.get("timestamp") or 0, x.get("transactionHash") or ""))
    return out


def write_outputs(rows: list[dict[str, Any]], csv_path: str, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fields = [
        "timestamp",
        "datetime_utc",
        "side",
        "size",
        "price",
        "notional_usdc",
        "outcome",
        "title",
        "eventSlug",
        "slug",
        "conditionId",
        "asset",
        "transactionHash",
        "proxyWallet",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ts = int(row.get("timestamp") or 0)
            size = float(row.get("size") or row.get("amount") or 0.0)
            price = float(row.get("price") or 0.0)
            dt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else ""
            writer.writerow(
                {
                    "timestamp": ts,
                    "datetime_utc": dt,
                    "side": row.get("side", ""),
                    "size": size,
                    "price": price,
                    "notional_usdc": size * price,
                    "outcome": row.get("outcome", ""),
                    "title": row.get("title", ""),
                    "eventSlug": row.get("eventSlug", ""),
                    "slug": row.get("slug", ""),
                    "conditionId": row.get("conditionId", ""),
                    "asset": row.get("asset", ""),
                    "transactionHash": row.get("transactionHash", ""),
                    "proxyWallet": row.get("proxyWallet", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default="Sovereign2013",
        help="Polymarket handle (default) or wallet address.",
    )
    parser.add_argument(
        "--endpoint",
        default="trades",
        choices=["trades", "activity"],
        help="Data API endpoint to use. `trades` is default; `activity` can be used as fallback.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Page size (recommended <=500).")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between pages.")
    parser.add_argument("--start", type=int, default=None, help="Unix timestamp lower bound.")
    parser.add_argument("--end", type=int, default=None, help="Unix timestamp upper bound.")
    parser.add_argument("--out", default="sovereign2013_trades.csv", help="Output CSV path.")
    parser.add_argument("--json-out", default="sovereign2013_trades.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        user_address = resolve_profile_address(args.user)
        rows = fetch_all_records(
            endpoint=args.endpoint,
            user_address=user_address,
            limit=args.limit,
            sleep_s=args.sleep,
            start=args.start,
            end=args.end,
        )
        write_outputs(rows, csv_path=args.out, json_path=args.json_out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Resolved user: {args.user} -> {user_address}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Fetched rows: {len(rows)}")
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
