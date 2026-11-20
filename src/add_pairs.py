"""Add candidate remake pairs to data/remake_pairs.csv (only if both films exist exactly once)."""
import pandas as pd

from src.search import load_data

# (original title, original year, remake title, remake year). Compiled from general
# knowledge, not independently verified; the script checks existence in the data only.
CANDIDATES = [
    ("Nayakan", 1987, "Dayavan", 1988),
    ("Thevar Magan", 1992, "Virasat", 1997),
    ("Alaipayuthey", 2000, "Saathiya", 2002),
    ("Kahaani", 2012, "Anamika", 2014),
    ("Ramanaa", 2002, "Gabbar Is Back", 2015),
    ("Thuppakki", 2012, "Holiday: A Soldier Is Never Off Duty", 2014),
    ("Mouna Guru", 2011, "Akira", 2016),
    ("Vikramarkudu", 2006, "Rowdy Rathore", 2012),
    ("Dabangg", 2010, "Gabbar Singh", 2012),
    ("3 Idiots", 2009, "Nanban", 2012),
    ("Don", 1978, "Billa", 1980),
    ("Don", 1978, "Don: The Chase Begins Again", 2006),
    ("Thani Oruvan", 2015, "Dhruva", 2016),
    ("Premam", 2015, "Premam", 2016),
    ("Bangalore Days", 2014, "Bangalore Naatkal", 2016),
    ("Mrs. Doubtfire", 1993, "Avvai Shanmughi", 1996),
    ("Avvai Shanmughi", 1996, "Chachi 420", 1997),
    ("Hitch", 2005, "Partner", 2007),
    ("Reservoir Dogs", 1992, "Kaante", 2002),
    ("Unfaithful", 2002, "Murder", 2004),
    ("Gol Maal", 1979, "Bol Bachchan", 2012),
    ("Himmatwala", 1983, "Himmatwala", 2013),
    ("Agneepath", 1990, "Agneepath", 2012),
    ("Okkadu", 2003, "Tevar", 2015),
    ("Okkadu", 2003, "Ghilli", 2004),
    ("Manichitrathazhu", 1993, "Bhool Bhulaiyaa", 2007),
    ("Manichitrathazhu", 1993, "Apthamitra", 2004),
    ("Apthamitra", 2004, "Chandramukhi", 2005),
]


def main():
    df, _ = load_data()
    lower = df["Title"].str.lower()

    def exact(title, year):
        hits = df[(lower == title.lower()) & (df["Release Year"] == year)]
        return hits["Title"].iloc[0] if len(hits) == 1 else None

    def suggest(title):
        first = title.split()[0].lower()
        hits = df[lower.str.contains(first, regex=False)].head(4)
        return "; ".join(f"{t} ({y})" for t, y in zip(hits["Title"], hits["Release Year"]))

    existing = pd.read_csv("data/remake_pairs.csv")
    new_rows, missing = [], []
    for ot, oy, rt, ry in CANDIDATES:
        a, b = exact(ot, oy), exact(rt, ry)
        if a and b:
            new_rows.append((a, oy, b, ry))
        else:
            if not a:
                missing.append(f"MISSING {ot} ({oy}) -> data has: {suggest(ot)}")
            if not b:
                missing.append(f"MISSING {rt} ({ry}) -> data has: {suggest(rt)}")

    all_pairs = pd.concat([existing, pd.DataFrame(new_rows, columns=existing.columns)])
    all_pairs = all_pairs.drop_duplicates().reset_index(drop=True)
    all_pairs.to_csv("data/remake_pairs.csv", index=False)
    print("pairs now:", len(all_pairs))
    print("\n".join(missing))


if __name__ == "__main__":
    main()
