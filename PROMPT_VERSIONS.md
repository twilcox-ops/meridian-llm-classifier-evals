| Version | Change | Category Acc | Urgency Acc | Cost/1,000 | Notes |
|---|---|---|---|---|---|
| v1 | Baseline minimal prompt | 100.0% | 41.4% | $1.0909 | Category solved; urgency near chance-level, needs work |
| v2 | Added per-category urgency rules (compliance=always medium; outage=high/critical split on person-in-danger+911 signal; sales/scheduling/billing=low/medium only), derived from cross-tabbing category x urgency on iteration data | 100.0% | 66.4% | $1.2098 | Category still 100%. Urgency jumped 41.4%→66.4%, entirely from compliance (41.4%→100%) and outage (→100%). Sales/scheduling/billing stayed 35-53% — consistent with earlier finding that low/medium split has no recoverable text signal (duplicate ticket text under different labels found in iteration data) |
