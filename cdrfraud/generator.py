"""Generate a week of synthetic call detail records (CDRs) with fraud planted in them.

Everything here is invented: numbers, cells and handsets. Nothing comes from a real
network. The generator is seeded, so the same seed always gives the same data.

What is planted, and why each variant exists:

| Fraud | Obvious variant (rules should catch) | Stealthy variant (built to slip past rules) |
|---|---|---|
| Wangiri | foreign number one-rings 250-400 subscribers in 2-3 hours | foreign number rings ~60 a day for 4-6 s, spread over the day |
| IRSF | hijacked line makes 30-120 long night calls to premium ranges | hijacked line makes 6-10 premium calls a day, spread out |
| SIM box | 16 SIMs on 4 shared handsets, one cell, 150-300 calls a day, nothing inbound | 10 SIMs, own handsets, two cells, 40-70 calls a day, a little inbound |

Hard negatives are planted too: busy business lines and diaspora callers who phone
abroad for a long time, so a detector can't score well by flagging "lots of calls".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-09-01")
DAYS = 7
CC = "263"  # subscribers are on a Zimbabwe-style numbering plan, all numbers invented

COMMON_FOREIGN = ["27", "44", "267", "260", "1"]  # South Africa, UK, Botswana, Zambia, US
HIGH_RISK_PREFIXES = ["882", "883", "675", "679"]  # satellite / international networks, PNG, Fiji

# Typical share of a subscriber's calls in each hour of the day.
HOUR_WEIGHTS = np.array([1, 0.5, 0.3, 0.3, 0.4, 1, 3, 6, 8, 8, 7, 7, 8, 7, 7, 7, 8, 9, 10, 10, 9, 7, 4, 2], float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()
BUSINESS_HOURS = np.array([0] * 7 + [4, 10, 10, 10, 9, 8, 10, 10, 10, 9, 5, 1] + [0] * 5, float) + 0.05
BUSINESS_HOURS /= BUSINESS_HOURS.sum()


@dataclass
class Dataset:
    cdrs: pd.DataFrame
    labels: pd.DataFrame  # number, fraud_type, variant: the planted fraud
    cohorts: pd.DataFrame  # number, cohort: legitimate lines that look unusual, for error analysis


class _Builder:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.rows: list[tuple] = []
        self.labels: list[tuple] = []
        self.cohorts: list[tuple] = []
        self.used: set[str] = set()

    # numbers and devices -------------------------------------------------------------
    def number(self, prefix: str, digits: int) -> str:
        while True:
            n = prefix + "".join(map(str, self.rng.integers(0, 10, digits)))
            if n not in self.used:
                self.used.add(n)
                return n

    def local(self) -> str:
        return self.number(CC + str(self.rng.choice(["71", "73", "77", "78"])), 7)

    def foreign(self, cc: str) -> str:
        return self.number(cc, 9)

    def imei(self) -> str:
        return "35" + "".join(map(str, self.rng.integers(0, 10, 13)))

    # calls ------------------------------------------------------------------------------
    def call(self, t, a, b, duration, cell=None, imei=None):
        answered = duration > 0
        self.rows.append((t, a, b, int(duration), answered, cell, imei))

    def times(self, day: int, n: int, weights=HOUR_WEIGHTS) -> list[pd.Timestamp]:
        hours = self.rng.choice(24, size=n, p=weights)
        offsets = day * 86400 + hours * 3600 + self.rng.integers(0, 3600, size=n)
        stamps = START.to_datetime64() + offsets.astype("timedelta64[s]")
        return [pd.Timestamp(t) for t in stamps]

    def talk_time(self, n: int, mean: float = 90) -> np.ndarray:
        d = self.rng.lognormal(np.log(mean), 0.9, n)
        d[self.rng.random(n) < 0.15] = 0  # unanswered
        return np.clip(d, 0, 3600)


def generate(seed: int = 7, n_subscribers: int = 3000) -> Dataset:
    b = _Builder(seed)
    rng = b.rng

    # Ordinary subscribers ---------------------------------------------------------------
    # Arrays, not lists: rng.choice on a list converts it to an array on every call.
    subs = np.array([b.local() for _ in range(n_subscribers)])
    off_net = np.array([b.local() for _ in range(1500)])  # numbers on other networks
    foreign_contacts = np.array([b.foreign(rng.choice(COMMON_FOREIGN)) for _ in range(400)])
    cells = np.array([f"HRE-{i:03d}" for i in range(120)])

    profile = {}
    for s in subs:
        contacts = list(rng.choice(subs, rng.integers(10, 40))) + list(rng.choice(off_net, rng.integers(5, 20)))
        if rng.random() < 0.25:
            contacts += list(rng.choice(foreign_contacts, rng.integers(1, 4)))
        profile[s] = {
            "rate": rng.gamma(2.0, 3.0),
            "contacts": np.array(contacts),
            "weights": None,
            "cells": list(rng.choice(cells, 3, replace=False)),
            "imei": b.imei(),
        }
        w = 1 / np.arange(1, len(contacts) + 1)
        profile[s]["weights"] = w / w.sum()

    def ordinary_day(s, day, rate_scale=1.0, weights=HOUR_WEIGHTS, mean_talk=90):
        p = profile[s]
        n = rng.poisson(p["rate"] * rate_scale)
        if not n:
            return
        dests = rng.choice(p["contacts"], size=n, p=p["weights"])
        for t, d, dur in zip(b.times(day, n, weights), dests, b.talk_time(n, mean_talk)):
            cell = p["cells"][0] if t.hour < 7 or t.hour > 19 else rng.choice(p["cells"])
            b.call(t, s, d, dur, cell, p["imei"])

    for day in range(DAYS):
        for s in subs:
            ordinary_day(s, day)
        # Calls in from other networks and from abroad.
        n = int(n_subscribers * 1.8)
        sources = np.where(rng.random(n) < 0.9, rng.choice(off_net, n), rng.choice(foreign_contacts, n))
        for t, src, dst, dur in zip(b.times(day, n), sources, rng.choice(subs, n), b.talk_time(n)):
            b.call(t, src, dst, dur)

    # Hard negatives -----------------------------------------------------------------
    business = list(rng.choice(subs, 25, replace=False))
    b.cohorts += [(s, "business line") for s in business]
    for s in business:
        profile[s]["rate"] = rng.uniform(50, 120)
        profile[s]["contacts"] = np.concatenate([profile[s]["contacts"], rng.choice(off_net, 150), rng.choice(subs, 150)])
        w = 1 / np.sqrt(np.arange(1, len(profile[s]["contacts"]) + 1))
        profile[s]["weights"] = w / w.sum()
        profile[s]["cells"] = list(rng.choice(cells, 5, replace=False))
        for day in range(DAYS):
            if day % 7 < 5:  # weekdays
                ordinary_day(s, day, weights=BUSINESS_HOURS, mean_talk=120)
        # Customers ring the business back.
        for day in range(DAYS):
            for t in b.times(day, int(profile[s]["rate"] * 0.4), BUSINESS_HOURS):
                b.call(t, rng.choice(off_net), s, b.talk_time(1, 120)[0])

    diaspora = [s for s in rng.choice(subs, 30, replace=False) if s not in business]
    b.cohorts += [(s, "diaspora caller") for s in diaspora]
    for s in diaspora:
        family = [b.foreign(rng.choice(["44", "27", "1"])) for _ in range(3)]
        for day in range(DAYS):
            for t in b.times(day, rng.integers(1, 5)):
                b.call(t, s, rng.choice(family), rng.uniform(600, 2400), profile[s]["cells"][0], profile[s]["imei"])

    # Wangiri ------------------------------------------------------------------------
    victims_by_caller = {}
    for i in range(3):
        caller = b.foreign(rng.choice(HIGH_RISK_PREFIXES))
        b.labels.append((caller, "wangiri", "obvious"))
        victims = []
        for _ in range(2):  # two campaigns
            day = int(rng.integers(0, DAYS))
            start = START + pd.Timedelta(days=day, hours=int(rng.integers(9, 18)))
            targets = rng.choice(subs, rng.integers(250, 400), replace=False)
            for v in targets:
                t = start + pd.Timedelta(seconds=int(rng.integers(0, 3 * 3600)))
                b.call(t, caller, v, 0 if rng.random() < 0.97 else rng.integers(1, 4))
            victims += [(v, start) for v in targets]
        victims_by_caller[caller] = victims

    stealth_caller = b.foreign(rng.choice(HIGH_RISK_PREFIXES))
    b.labels.append((stealth_caller, "wangiri", "stealthy"))
    for day in range(DAYS):
        for t, v in zip(b.times(day, 60), rng.choice(subs, 60, replace=False)):
            b.call(t, stealth_caller, v, rng.integers(4, 7) if rng.random() < 0.3 else 0)
    victims_by_caller[stealth_caller] = [(v, START) for v in rng.choice(subs, 200, replace=False)]

    # A few victims ring back: those calls are what earn the fraudster money.
    for caller, victims in victims_by_caller.items():
        for v, when in victims:
            if rng.random() < 0.03:
                b.cohorts.append((v, "Wangiri victim who called back"))
                t = when + pd.Timedelta(minutes=int(rng.integers(5, 600)))
                b.call(t, v, caller, rng.uniform(60, 400), profile[v]["cells"][0], profile[v]["imei"])

    # IRSF -----------------------------------------------------------------------------
    excluded = set(business) | set(diaspora)
    candidates = [s for s in subs if s not in excluded]
    hijacked = list(rng.choice(candidates, 11, replace=False))
    for s in hijacked[:8]:
        b.labels.append((s, "irsf", "obvious"))
        premium = [b.foreign(rng.choice(HIGH_RISK_PREFIXES)) for _ in range(rng.integers(3, 8))]
        for day in rng.choice(DAYS, rng.integers(1, 3), replace=False):
            base = START + pd.Timedelta(days=int(day))
            for _ in range(rng.integers(30, 120)):
                t = base + pd.Timedelta(seconds=int(rng.integers(0, 5 * 3600)))  # 00:00-05:00
                b.call(t, s, rng.choice(premium), rng.uniform(300, 1800), profile[s]["cells"][0], profile[s]["imei"])
    for s in hijacked[8:]:
        b.labels.append((s, "irsf", "stealthy"))
        premium = [b.foreign(rng.choice(HIGH_RISK_PREFIXES)) for _ in range(3)]
        for day in range(DAYS):
            for t in b.times(day, rng.integers(6, 11)):
                b.call(t, s, rng.choice(premium), rng.uniform(120, 600), profile[s]["cells"][0], profile[s]["imei"])

    # SIM boxes ------------------------------------------------------------------------
    destinations = np.concatenate([subs, off_net])
    for box in range(2):
        cell = rng.choice(cells)
        modules = [b.imei() for _ in range(4)]
        for _ in range(16):
            sim = b.local()
            b.labels.append((sim, "simbox", "obvious"))
            for day in range(DAYS):
                n = rng.integers(150, 300)
                for t, d, dur in zip(b.times(day, n), rng.choice(destinations, n), b.talk_time(n, 100)):
                    b.call(t, sim, d, dur, cell, rng.choice(modules))  # SIMs rotate across handsets

    stealth_cells = list(rng.choice(cells, 2, replace=False))
    for _ in range(10):
        sim, handset = b.local(), b.imei()
        b.labels.append((sim, "simbox", "stealthy"))
        for day in range(DAYS):
            n = rng.integers(40, 70)
            for t, d, dur in zip(b.times(day, n), rng.choice(destinations, n), b.talk_time(n, 100)):
                b.call(t, sim, d, dur, rng.choice(stealth_cells), handset)
            for t in b.times(day, rng.integers(1, 4)):
                b.call(t, rng.choice(off_net), sim, b.talk_time(1)[0])

    cdrs = pd.DataFrame(b.rows, columns=["start_time", "a_number", "b_number", "duration_s", "answered", "cell_id", "imei"])
    cdrs = cdrs.sort_values("start_time", kind="stable").reset_index(drop=True)
    cdrs.insert(0, "call_id", np.arange(len(cdrs)))
    labels = pd.DataFrame(b.labels, columns=["number", "fraud_type", "variant"])
    cohorts = pd.DataFrame(b.cohorts, columns=["number", "cohort"]).drop_duplicates("number")
    # A hijacked line can also ring back a Wangiri caller; it belongs with the fraud.
    cohorts = cohorts[~cohorts["number"].isin(labels["number"])].reset_index(drop=True)
    return Dataset(cdrs, labels, cohorts)
