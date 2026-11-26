"""Offline look-alikes of the book's tabular, collaborative-filtering and text datasets.

    BULLDOZERS     bluebook-for-bulldozers/TrainAndValid.csv, Test.csv  (Kaggle auction prices, Ch. 9)
    ADULT_SAMPLE   adult_sample/adult.csv                               (census income, Ch. 1)
    ML_100k        ml-100k/u.data, u.item                               (MovieLens ratings, Ch. 8)
    ML_SAMPLE      movie_lens_sample/ratings.csv                        (small MovieLens, Ch. 1)
    IMDB           imdb/{train,test}/{pos,neg}/*.txt, imdb/unsup/*.txt  (movie reviews, Ch. 1, 10)
    WIKITEXT       wikitext_synth/train.txt                             (general text to pre-train a LM)
    HUMAN_NUMBERS  human_numbers/{train,valid}.txt                      (exactly the real dataset, Ch. 12)

The relationships inside the data are planted on purpose (and documented below), so that the
chapter analyses find something real: important and redundant columns, a time trend that
makes extrapolation hard, users and movies with interpretable tastes, sentiment-bearing words.
"""

import numpy as np
import pandas as pd

# ==================================================================================================
# Bulldozers (Chapter 9)
# ==================================================================================================
_STATES = ["Alabama", "Arizona", "California", "Colorado", "Florida", "Georgia", "Illinois", "Texas",
           "Ohio", "New York", "North Carolina", "Washington", "Oklahoma", "Pennsylvania", "Unspecified"]
_GROUPS = {  # ProductGroup: (ProductGroupDesc, class description, base price, typical sizes)
    "TEX": ("Track Excavators", "Hydraulic Excavator, Track", 45000, ["Large", "Medium", "Small", "Mini"]),
    "WL": ("Wheel Loader", "Wheel Loader", 38000, ["Large", "Medium", "Small"]),
    "TTT": ("Track Type Tractors", "Track Type Tractor, Dozer", 42000, ["Large", "Medium", "Small"]),
    "BL": ("Backhoe Loaders", "Backhoe Loader", 22000, ["Mini", "Small"]),
    "SSL": ("Skid Steer Loaders", "Skid Steer Loader", 11000, ["Compact", "Mini"]),
    "MG": ("Motor Graders", "Motorgrader", 60000, ["Large", "Medium"]),
}
_SIZE_MULT = {"Large": 1.9, "Large / Medium": 1.5, "Medium": 1.2, "Small": 0.85, "Mini": 0.6, "Compact": 0.5}
_HP = {"Large": "250.0 to 500.0", "Large / Medium": "175.0 to 225.0", "Medium": "150.0 to 175.0",
       "Small": "100.0 to 120.0", "Mini": "40.0 to 60.0", "Compact": "1601.0 to 1751.0 Lb Operating Capacity"}


def only_where(mask, values):
    """values where mask is True, missing (NaN) elsewhere (works for strings too)."""
    s = pd.Series(np.asarray(values), dtype=object)
    s[~np.asarray(mask, dtype=bool)] = np.nan
    return s.values


def make_bulldozers(root, n=60000, seed=10):
    """TrainAndValid.csv (sales 1989-2012/04) and Test.csv (2012/05-2012/11, no prices).

    Planted structure:
      * price depends mostly on the model (ModelID / fiModelDesc / ProductSize / fiProductClassDesc),
        the machine's age (saleYear - YearMade), Enclosure and Coupler_System (for skid steers);
      * prices drift upwards over time (inflation), so later sales are pricier: random forests can't
        extrapolate this trend, but neural nets can;
      * redundant pairs: ProductGroup/ProductGroupDesc, fiModelDesc/fiBaseModel, saleYear/saleElapsed,
        Hydraulics_Flow/Grouser_Tracks/Coupler_System (missing together, as in the real data);
      * YearMade is 1000 (an unknown-year code) for ~8% of rows; most hours-meter readings are missing;
      * SalesID and MachineID are IDs that grow over time (useful to find out-of-domain data).
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    # ---- the catalogue of models ----
    models = []
    for mid in range(1, 401):
        group = list(_GROUPS)[rng.integers(len(_GROUPS))]
        desc, cls, base, sizes = _GROUPS[group]
        size = sizes[rng.integers(len(sizes))]
        base_model = f"{rng.choice(list('DLEGHKS'))}{rng.integers(1, 99) * 10}"
        secondary = ["B", "C", "D", "G", "LC", "H", None][rng.integers(7)]
        series = ["II", "III", "-6", "LT", None, None, None][rng.integers(7)]
        model_desc = f"{base_model}{secondary or ''}"
        models.append(dict(ModelID=mid * 7 + 1000, ProductGroup=group, ProductGroupDesc=desc, ProductSize=size,
                           fiBaseModel=base_model, fiSecondaryDesc=secondary, fiModelSeries=series,
                           fiModelDesc=model_desc, fiProductClassDesc=f"{cls} - {_HP[size]} Horsepower",
                           value=base * _SIZE_MULT[size] * rng.lognormal(0, 0.35),
                           popularity=rng.pareto(1.2) + 0.1))
    pop = np.array([m["popularity"] for m in models])
    midx = rng.choice(len(models), size=n, p=pop / pop.sum())
    # ---- sale dates: more auctions every year ----
    start, end = np.datetime64("1989-01-17"), np.datetime64("2012-04-28")
    span = int((end - start).astype(int))
    days = np.sort((rng.power(1.8, n) * span).astype(int))
    saledate = start + days.astype("timedelta64[D]")
    sale_year = saledate.astype("datetime64[Y]").astype(int) + 1970
    age = np.clip(rng.gamma(2.0, 4.0, n).round(), 0, 40).astype(int)
    year_made = sale_year - age
    rows = {k: [models[i][k] for i in midx] for k in models[0] if k not in ("value", "popularity")}
    df = pd.DataFrame(rows)
    df.insert(0, "SalesID", 1139246 + np.arange(n) * 7 + rng.integers(0, 7, n))
    df["MachineID"] = (999089 + days * 3 + rng.integers(0, 400000, n)).astype(int)
    df["datasource"] = np.select([sale_year < 2000, sale_year < 2006, sale_year < 2010], [121, 132, 136], 149)
    df["auctioneerID"] = rng.choice([1, 2, 3, 4, 5, 6, 7, 99, np.nan], n, p=[.35, .15, .1, .1, .08, .07, .05, .05, .05])
    df["YearMade"] = np.where(rng.random(n) < 0.08, 1000, year_made)
    hours = np.where(rng.random(n) < 0.65, np.nan, np.maximum(0, age * rng.normal(900, 300, n)))
    hours = np.where(rng.random(n) < 0.1, 0, hours)
    df["MachineHoursCurrentMeter"] = hours
    df["UsageBand"] = only_where(~(np.isnan(hours) | (hours == 0)),
                                 np.select([hours < 3000, hours < 9000], ["Low", "Medium"], "High"))
    df["saledate"] = pd.to_datetime(saledate).strftime("%m/%d/%Y 0:00")
    df["state"] = rng.choice(_STATES, n)
    is_ssl = df.ProductGroup.values == "SSL"
    df["Enclosure"] = rng.choice(["OROPS", "EROPS", "EROPS w AC"], n, p=[.45, .3, .25])
    df["Drive_System"] = only_where(df.ProductGroup.isin(["BL", "MG"]), rng.choice(["Two Wheel Drive", "Four Wheel Drive"], n))
    df["Hydraulics"] = only_where(rng.random(n) < 0.8, rng.choice(["Standard", "2 Valve", "Auxiliary"], n))
    df["Coupler_System"] = only_where(is_ssl, rng.choice(["None or Unspecified", "Yes"], n, p=[0.6, 0.4]))
    df["Grouser_Tracks"] = only_where(is_ssl, np.full(n, "None or Unspecified"))
    df["Hydraulics_Flow"] = only_where(is_ssl, rng.choice(["Standard", "High Flow"], n, p=[.85, .15]))
    df["Tire_Size"] = only_where(df.ProductGroup.isin(["WL", "MG"]), rng.choice(["17.5", "20.5", "23.5", "None or Unspecified"], n))
    df["Stick_Length"] = only_where(df.ProductGroup == "TEX", rng.choice(["9' 6\"", "10' 6\"", "None or Unspecified"], n))
    # ---- the price ----
    value = np.array([models[i]["value"] for i in midx])
    price = value * np.exp(-0.075 * age)                                       # depreciation
    price *= 1 + 0.035 * (sale_year - 1989) + 0.05 * np.maximum(0, sale_year - 2008)   # inflation trend
    price *= np.select([df.Enclosure == "EROPS w AC", df.Enclosure == "EROPS"], [1.3, 1.12], 1.0)
    price *= np.where(df.Coupler_System.values == "Yes", 1.35, 1.0)
    price *= np.where(df.Hydraulics_Flow.values == "High Flow", 1.2, 1.0)
    price *= np.where(np.nan_to_num(hours) > 8000, 0.85, 1.0)
    price *= rng.lognormal(0, 0.18, n)                                          # auction noise
    df["SalePrice"] = np.clip(price, 4750, 142000).round(-1)
    df.to_csv(root / "TrainAndValid.csv", index=False)
    # ---- Test.csv: the six months after the training data ends ----
    nt = n // 20
    tdf = df.sample(nt, random_state=seed).drop(columns="SalePrice").reset_index(drop=True)
    tdays = np.sort(rng.integers(0, 200, nt))
    tdate = np.datetime64("2012-05-01") + tdays.astype("timedelta64[D]")
    tdf["saledate"] = pd.to_datetime(tdate).strftime("%m/%d/%Y 0:00")
    tdf["SalesID"] = df.SalesID.max() + 1 + np.arange(nt) * 7
    tdf["YearMade"] = np.where(tdf.YearMade == 1000, 1000, tdf.YearMade + 1)
    tdf.to_csv(root / "Test.csv", index=False)


# ==================================================================================================
# Adult census income (Chapter 1)
# ==================================================================================================
def make_adult(root, n=10000, seed=11):
    """adult.csv: predict whether someone earns >=50k (values keep the real file's leading spaces)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    edu = [("HS-grad", 9), ("Some-college", 10), ("Bachelors", 13), ("Masters", 14), ("Doctorate", 16),
           ("Assoc-voc", 11), ("11th", 7), ("Prof-school", 15)]
    e = rng.choice(len(edu), n, p=[.32, .22, .17, .06, .02, .05, .1, .06])
    age = rng.integers(17, 80, n)
    occupations = ["Prof-specialty", "Exec-managerial", "Craft-repair", "Sales", "Adm-clerical",
                   "Other-service", "Machine-op-inspct", "Tech-support"]
    occ = rng.integers(0, len(occupations), n)
    married = rng.random(n) < np.clip((age - 20) / 30, 0.05, 0.65)
    hours = np.clip(rng.normal(40, 11, n), 5, 99).round()
    gain = np.where(rng.random(n) < 0.08, rng.choice([2174, 5178, 7688, 15024], n), 0)
    logit = (-9.5 + 0.35 * np.array([edu[i][1] for i in e]) + 0.04 * np.minimum(age, 60) + 1.6 * married
             + 0.03 * hours + 0.8 * np.isin(occ, [0, 1]) + 2.5 * (gain > 5000))
    rich = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "age": age, "workclass": rng.choice([" Private", " Self-emp-not-inc", " Local-gov", " State-gov", " ?"], n, p=[.72, .08, .08, .07, .05]),
        "fnlwgt": rng.integers(20000, 600000, n), "education": [" " + edu[i][0] for i in e],
        "education-num": [float(edu[i][1]) for i in e],
        "marital-status": np.where(married, " Married-civ-spouse", rng.choice([" Never-married", " Divorced"], n)),
        "occupation": [" " + occupations[i] for i in occ],
        "relationship": np.where(married, rng.choice([" Husband", " Wife"], n, p=[.85, .15]), rng.choice([" Not-in-family", " Own-child", " Unmarried"], n)),
        "race": rng.choice([" White", " Black", " Asian-Pac-Islander", " Other"], n, p=[.85, .09, .04, .02]),
        "sex": rng.choice([" Male", " Female"], n, p=[.67, .33]), "capital-gain": gain,
        "capital-loss": np.where(rng.random(n) < 0.04, 1902, 0), "hours-per-week": hours,
        "native-country": rng.choice([" United-States", " Mexico", " ?", " Philippines"], n, p=[.9, .04, .03, .03]),
        "salary": np.where(rich, ">=50k", "<50k")})
    for col, frac in (("education-num", 0.01), ("occupation", 0.02)):        # some missing values
        df.loc[rng.random(n) < frac, col] = np.nan
    df.to_csv(root / "adult.csv", index=False)


# ==================================================================================================
# MovieLens (Chapters 1 and 8)
# ==================================================================================================
_GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
           "War", "Western"]
# Each "style" of film has its own title words, genres and position in a 3-D taste space:
#   axis 0 = action/sci-fi (+) vs romance/drama (-), axis 1 = mainstream (+) vs arthouse (-), axis 2 = dark (+) vs light (-)
_STYLES = {
    "scifi": (["Star", "Galaxy", "Robot", "Planet", "Alien", "Orbit", "Cyborg"], ["Sci-Fi", "Action"], (1.2, 0.8, 0.3)),
    "action": (["Fury", "Strike", "Bullet", "Mission", "Thunder", "Speed"], ["Action", "Thriller"], (1.0, 1.0, 0.4)),
    "romance": (["Love", "Heart", "Kiss", "Wedding", "Valentine", "Promise"], ["Romance", "Comedy"], (-1.1, 0.7, -0.6)),
    "drama": (["River", "Silence", "Winter", "Letters", "Garden", "Memory"], ["Drama"], (-0.8, -0.9, 0.2)),
    "horror": (["Blood", "Scream", "Night", "Grave", "Curse", "Shadow"], ["Horror", "Thriller"], (0.5, -0.2, 1.4)),
    "kids": (["Puppy", "Toy", "Dragon", "Pirate", "Magic", "Bunny"], ["Children's", "Animation", "Comedy"], (0.0, 1.1, -1.3)),
    "noir": (["Confidential", "Detective", "Smoke", "Alley", "Rain", "Suspect"], ["Crime", "Film-Noir", "Mystery"], (0.2, -1.2, 0.9)),
}
_NOUNS = ["Return", "Night", "City", "Story", "Road", "Game", "Kingdom", "Secret", "Day", "Island", "House", "Legend"]


def make_movielens(root, n_users=943, n_movies=1682, n_ratings=100000, seed=12):
    """ml-100k/u.data (user, movie, rating, timestamp; tab-separated) and u.item (movie titles + genres).

    Each movie has a style (sci-fi, romance, ...) that sets its title words, genres and its hidden
    taste vector; each user has a taste vector; rating ~ user . movie + movie quality + user bias.
    Some movies are excellent (high bias), a few are widely seen but poorly rated.
    """
    rng = np.random.default_rng(seed)
    styles = list(_STYLES)
    m_style = rng.integers(0, len(styles), n_movies)
    m_vec = np.array([_STYLES[styles[s]][2] for s in m_style]) + rng.normal(0, 0.35, (n_movies, 3))
    m_bias = rng.normal(0, 0.5, n_movies)
    m_pop = rng.pareto(1.1, n_movies) + 0.05 + 0.3 * (m_vec[:, 1] > 0)
    u_vec = rng.normal(0, 0.8, (n_users, 3))
    u_bias = rng.normal(0, 0.35, n_users)
    u_act = rng.pareto(1.3, n_users) + 0.2
    titles, used = [], set()
    for i in range(n_movies):
        words, _, _ = _STYLES[styles[m_style[i]]]
        while True:
            w = rng.choice(words)
            t = rng.choice([f"{w} {rng.choice(_NOUNS)}", f"The {w}", f"{rng.choice(_NOUNS)} of the {w}",
                            f"{w} {rng.choice(_NOUNS)} {rng.integers(2, 4)}"])
            year = int(rng.integers(1930, 1998) if rng.random() < 0.3 else rng.integers(1990, 1998))
            title = f"{t} ({year})"
            if title not in used:
                used.add(title)
                titles.append(title)
                break
    # sample (user, movie) pairs proportional to activity x popularity, without duplicates
    pairs = set()
    while len(pairs) < n_ratings:
        u = rng.choice(n_users, n_ratings, p=u_act / u_act.sum())
        m = rng.choice(n_movies, n_ratings, p=m_pop / m_pop.sum())
        pairs.update(zip(u.tolist(), m.tolist()))
    pairs = list(pairs)[:n_ratings]
    u, m = np.array(pairs).T
    score = 3.5 + (u_vec[u] * m_vec[m]).sum(1) * 0.6 + m_bias[m] + u_bias[u] + rng.normal(0, 0.6, len(u))
    rating = np.clip(np.round(score), 1, 5).astype(int)
    ts = rng.integers(874724710, 893286638, len(u))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "u.data", "w") as f:
        for a, b, r, t in zip(u + 1, m + 1, rating, ts):
            f.write(f"{a}\t{b}\t{r}\t{t}\n")
    with open(root / "u.item", "w", encoding="latin-1") as f:
        for i, title in enumerate(titles):
            g = set(_STYLES[styles[m_style[i]]][1])
            flags = "|".join("1" if name in g else "0" for name in _GENRES)
            f.write(f"{i + 1}|{title}|01-Jan-{title[-5:-1]}||http://example.com/{i + 1}|{flags}\n")
    with open(root / "u.user", "w") as f:
        for i in range(n_users):
            f.write(f"{i + 1}|{rng.integers(15, 70)}|{rng.choice(['M', 'F'])}|student|00000\n")
    with open(root / "README", "w") as f:
        f.write("Synthetic MovieLens-100k look-alike: 943 users, 1682 movies, 100000 ratings.\n")


def make_ml_sample(root, ml_root):
    """movie_lens_sample/ratings.csv: the 100 most active users x 100 most rated movies."""
    import pandas as pd
    r = pd.read_csv(ml_root / "u.data", delimiter="\t", header=None, names=["userId", "movieId", "rating", "timestamp"])
    top_u = r.userId.value_counts().index[:100]
    top_m = r.movieId.value_counts().index[:100]
    r[r.userId.isin(top_u) & r.movieId.isin(top_m)].to_csv(root / "ratings.csv", index=False)


# ==================================================================================================
# Movie reviews (IMDb) and general text (Chapters 1 and 10)
# ==================================================================================================
_POS = ["great", "wonderful", "brilliant", "excellent", "moving", "beautiful", "superb", "funny", "clever",
        "touching", "gripping", "charming", "fantastic", "memorable"]
_NEG = ["terrible", "boring", "awful", "dull", "stupid", "painful", "predictable", "silly", "weak",
        "pointless", "lazy", "forgettable", "confusing", "bad"]
_THINGS = ["acting", "plot", "script", "story", "soundtrack", "ending", "dialogue", "direction",
           "cinematography", "cast", "pacing", "special effects", "villain", "humour"]
_PEOPLE = ["the director", "the lead actor", "the lead actress", "the writer", "the whole cast",
           "the composer", "the young star", "the producer"]
_FILMS = ["this movie", "this film", "the film", "this picture", "the sequel", "this remake", "the story"]
_NEUTRAL = ["I watched it {when} with {who}.", "It is about {topic}.", "The film runs for about {n} minutes.",
            "It was released in {year}.", "I saw it at {where}.", "The book it is based on is about {topic}.",
            "{Person} also made {film2} a few years ago."]
_WHEN = ["last night", "on Sunday", "on a long flight", "at the weekend", "years ago", "twice"]
_WHO = ["my family", "friends", "my sister", "my kids", "nobody", "my parents"]
_TOPICS = ["a family on a farm", "a detective in Paris", "two brothers at war", "a robot who learns to love",
           "a school for wizards", "a band on tour", "a lost dog", "a bank robbery", "a ship in a storm"]
_WHERE = ["the cinema", "a festival", "home", "a friend's house"]


def _sentence(rng, sentiment):
    """One sentence. sentiment: +1 positive, -1 negative, 0 neutral."""
    if sentiment == 0:
        s = rng.choice(_NEUTRAL)
        return s.format(when=rng.choice(_WHEN), who=rng.choice(_WHO), topic=rng.choice(_TOPICS),
                        n=rng.integers(85, 180), year=rng.integers(1950, 2020), where=rng.choice(_WHERE),
                        Person=rng.choice(_PEOPLE).capitalize(), film2=rng.choice(["another film", "a TV show"]))
    good, bad = (_POS, _NEG) if sentiment > 0 else (_NEG, _POS)
    w = rng.choice(good)
    negated = rng.random() < 0.15                 # "not bad" means good: the model has to read context
    if negated:
        w = "not " + rng.choice(bad)
    t = rng.integers(8)
    if t == 0:
        s = f"The {rng.choice(_THINGS)} was {w}."
    elif t == 1:
        s = f"{rng.choice(_FILMS).capitalize()} is {rng.choice(['really', 'truly', 'just', 'so', 'quite'])} {w}."
    elif t == 2:
        s = f"{rng.choice(_PEOPLE).capitalize()} did a {w} job."
    elif t == 3:
        s = f"I thought the {rng.choice(_THINGS)} was {w} and the {rng.choice(_THINGS)} was {rng.choice(good)}."
    elif t == 4:
        s = (f"{rng.choice(['I loved it', 'Highly recommended', 'A must see', 'Ten out of ten', 'I will watch it again'])}"
             if sentiment > 0 else
             f"{rng.choice(['I hated it', 'Avoid it', 'What a waste of time', 'One out of ten', 'I walked out'])}")
        s += rng.choice([".", "!", "!!!", "!!!!!!"])
    elif t == 5:
        s = f"It is one of the most {w} films I have ever seen."
    elif t == 6:
        s = f"{rng.choice(['Honestly', 'Overall', 'In the end', 'Sadly', 'Luckily'])}, the {rng.choice(_THINGS)} is {w}."
    else:
        s = f"{w.upper() if rng.random() < 0.5 else w.capitalize()} {rng.choice(['!', '.'])}"
    return s


def _review(rng, label):
    """A review of 4-12 sentences: mostly on-topic sentiment, some neutral, some mixed."""
    n = rng.integers(4, 13)
    sents = []
    for _ in range(n):
        r = rng.random()
        sents.append(_sentence(rng, 0 if r < 0.3 else (label if r < 0.85 else -label)))
    text = " ".join(sents)
    if rng.random() < 0.3:                         # IMDb reviews contain HTML line breaks
        k = rng.integers(1, len(sents))
        text = " ".join(sents[:k]) + "<br /><br />" + " ".join(sents[k:])
    return text


def make_imdb(root, n_train=6000, n_test=2000, n_unsup=6000, seed=13):
    """imdb/{train,test}/{pos,neg}/<id>_<stars>.txt and imdb/unsup/<id>_0.txt."""
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("test", n_test)):
        for lbl, sign in (("pos", 1), ("neg", -1)):
            out = root / split / lbl
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n // 2):
                stars = rng.integers(7, 11) if sign > 0 else rng.integers(1, 5)
                (out / f"{i}_{stars}.txt").write_text(_review(rng, sign))
    out = root / "unsup"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_unsup):
        (out / f"{i}_0.txt").write_text(_review(rng, rng.choice([-1, 1])))
    (root / "README").write_text("Synthetic IMDb look-alike (pos/neg movie reviews + unlabelled reviews).\n")


def make_wikitext(root, n_articles=2000, seed=14):
    """General English-like text (encyclopedia style) to pre-train a language model (ULMFiT step 1).
    Like real articles, each one keeps a consistent tone (praise or criticism), so a language model
    trained on it learns that 'great' and 'superb' go together, as do 'dull' and 'boring'."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for _ in range(n_articles):
        tone = int(rng.choice([-1, 1]))
        good, bad = (_POS, _NEG) if tone > 0 else (_NEG, _POS)
        word = lambda: rng.choice(good) if rng.random() < 0.85 else rng.choice(bad)  # noqa: E731
        lines.append(f"= {rng.choice(_FILMS).capitalize()} ( {rng.integers(1920, 2020)} film ) =")
        for _ in range(rng.integers(12, 25)):
            t = rng.integers(5)
            if t == 0:
                s = f"The film is about {rng.choice(_TOPICS)} and was released in {rng.integers(1920, 2020)} ."
            elif t == 1:
                s = f"{rng.choice(_PEOPLE).capitalize()} said that the {rng.choice(_THINGS)} was {word()} ."
            elif t == 2:
                s = f"It was shown at {rng.choice(_WHERE)} and critics called it {word()} ."
            elif t == 3:
                s = f"In {rng.integers(1900, 2020)} , {rng.choice(_PEOPLE)} moved to {rng.choice(['London', 'Paris', 'Tokyo', 'New York', 'Rome'])} ."
            else:
                s = _sentence(rng, tone if rng.random() < 0.85 else -tone)
            lines.append(s)
    (root / "train.txt").write_text("\n".join(lines) + "\n")


# ==================================================================================================
# Human numbers (Chapter 12) - this one is generated exactly like the real dataset
# ==================================================================================================
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n):
    """8023 -> 'eight thousand twenty three'."""
    words = []
    if n >= 1000:
        words += [_ONES[n // 1000], "thousand"]
        n %= 1000
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n > 0:
        words.append(_ONES[n])
    return " ".join(words)


def make_human_numbers(root):
    """human_numbers/train.txt (one .. seven thousand nine hundred ninety nine) and valid.txt (8000..9999)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.txt").write_text("".join(number_to_words(i) + " \n" for i in range(1, 8000)))
    (root / "valid.txt").write_text("".join(number_to_words(i) + " \n" for i in range(8000, 10000)))
