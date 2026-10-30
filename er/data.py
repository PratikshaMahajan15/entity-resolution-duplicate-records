"""Data loading. Both loaders return the same schema:
given_name, surname, address, suburb, postcode, date_of_birth, entity_id
(entity_id = ground-truth identity; records sharing it are duplicates)."""
import random

import pandas as pd

FIELDS = ["given_name", "surname", "address", "suburb", "postcode", "date_of_birth"]


# --------------------------------------------------------------- real benchmark
def load_febrl3() -> pd.DataFrame:
    """Febrl3: 5,000 records, 2,000 originals + 3,000 duplicates, with realistic noise.
    The recordlinkage package ships the file, so no download is needed."""
    from recordlinkage.datasets import load_febrl3 as _load

    raw = _load()
    addr = (raw["street_number"].fillna("").astype(str).str.strip() + " "
            + raw["address_1"].fillna("") + " " + raw["address_2"].fillna(""))
    df = pd.DataFrame({
        "given_name": raw["given_name"],
        "surname": raw["surname"],
        "address": addr.str.replace(r"\s+", " ", regex=True).str.strip(),
        "suburb": raw["suburb"],
        "postcode": raw["postcode"],
        "date_of_birth": raw["date_of_birth"],
    })
    # ids look like "rec-1496-org" / "rec-1496-dup-0"; the number is the entity
    df["entity_id"] = raw.index.to_series().str.extract(r"rec-(\d+)-")[0].astype(int).to_numpy()
    return df.reset_index(drop=True)


# -------------------------------------------------------------- synthetic data
def _typo(s: str, rng: random.Random, n: int = 1) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(n):
        if len(s) < 3:
            return s
        i = rng.randrange(len(s) - 1)
        op = rng.choice(["delete", "swap", "substitute", "insert"])
        if op == "delete":
            s = s[:i] + s[i + 1:]
        elif op == "swap":
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        elif op == "substitute":
            s = s[:i] + rng.choice(letters) + s[i + 1:]
        else:
            s = s[:i] + rng.choice(letters) + s[i:]
    return s


def _corrupt_dob(dob: str, rng: random.Random) -> str:
    if rng.random() < 0.5:                      # day/month swap
        return dob[:4] + dob[6:8] + dob[4:6]
    i = rng.randrange(8)                        # one wrong digit
    return dob[:i] + rng.choice("0123456789") + dob[i + 1:]


def _corrupt(rec: dict, rng: random.Random) -> dict:
    r = dict(rec)
    if rng.random() < 0.30:
        r["given_name"] = _typo(r["given_name"], rng)
    if rng.random() < 0.30:
        r["surname"] = _typo(r["surname"], rng)
    if rng.random() < 0.05:                     # first/last name swapped
        r["given_name"], r["surname"] = r["surname"], r["given_name"]
    if rng.random() < 0.40:
        r["address"] = _typo(r["address"], rng, rng.choice([1, 2, 3]))
    if rng.random() < 0.10:
        r["suburb"] = _typo(r["suburb"], rng)
    if rng.random() < 0.15:
        r["date_of_birth"] = _corrupt_dob(r["date_of_birth"], rng)
    if rng.random() < 0.08:                     # breaks pincode blocking on purpose
        p = r["postcode"]
        i = rng.randrange(len(p))
        r["postcode"] = p[:i] + rng.choice("0123456789") + p[i + 1:]
    if rng.random() < 0.05:
        r["suburb"] = ""                        # missing value
    return r


def generate_synthetic(n=5000, dup_rate=0.20, household_rate=0.04,
                       n_pincodes=300, seed=42) -> pd.DataFrame:
    """Synthetic customers with planted duplicates and hard negatives
    (relatives sharing surname + address but being different people)."""
    from faker import Faker

    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("en_IN")
    pins = [fake.postcode() for _ in range(n_pincodes)]

    n_dup, n_house = int(n * dup_rate), int(n * household_rate)
    n_base = n - n_dup - n_house

    def dob():
        return fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y%m%d")

    base = [{
        "given_name": fake.first_name(), "surname": fake.last_name(),
        "address": fake.street_address().replace("\n", ", "),
        "suburb": fake.city(), "postcode": rng.choice(pins),
        "date_of_birth": dob(), "entity_id": e,
    } for e in range(n_base)]

    rows = list(base)
    for k, src in enumerate(rng.sample(base, n_house)):          # hard negatives
        rec = dict(src)
        rec.update(given_name=fake.first_name(), date_of_birth=dob(), entity_id=n_base + k)
        rows.append(rec)
    for src in rng.sample(base, n_dup):                           # duplicates
        rows.append(_corrupt(src, rng))

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
