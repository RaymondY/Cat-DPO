"""Build K pseudo-categories WITHOUT using any harm labels.

Cat-DPO needs a category assignment C(x), but nothing in the method requires that
assignment to come from a curated taxonomy. This script discards PKU-SafeRLHF's 19
annotated harm categories entirely and derives K categories by unsupervised clustering
of the training prompts (TF-IDF -> truncated SVD -> k-means, all local; no model and no
label is consulted at any point).

Training then uses only these discovered categories, via
    train_catdpo.py --category_map <this script's output>
while evaluation still reports the original PKU 19 categories, so the comparison against
label-supervised runs is unchanged.

Writes {prompt_md5_12: [cluster_id]} for the train and test prompts.
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

from data import load_pairs, HARM_CATEGORIES


def ph(p):
    return hashlib.md5(p.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description="Label-free category construction for Cat-DPO")
    ap.add_argument("--k", type=int, default=19,
                    help="number of clusters; 19 keeps the number of dual variables unchanged")
    ap.add_argument("--subset", default="alpaca2-7b")
    ap.add_argument("--data_mode", default="bettersafe")
    ap.add_argument("--out", default="outputs/cluster_categories_k19.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    train = load_pairs(args.data_mode, "train", subset=args.subset)
    test = load_pairs(args.data_mode, "test", subset=args.subset)
    # Cluster only the pairs that carry a margin (safe-unsafe); safe-safe pairs stay
    # category-free, exactly as they do under the annotated taxonomy.
    tr_lab = [ex for ex in train if ex["category_indices"]]
    te_lab = [ex for ex in test if ex["category_indices"]]
    print(f"train labeled pairs {len(tr_lab)}/{len(train)}, test {len(te_lab)}/{len(test)}")

    # Feature = prompt + rejected response, i.e. the text the harm annotation describes.
    # No harm label enters the features.
    def text(ex):
        return ex["prompt"] + " \n " + ex["rejected"]

    vec = TfidfVectorizer(max_features=50000, stop_words="english",
                          ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform([text(ex) for ex in tr_lab])
    print("tfidf:", X.shape)
    reducer = make_pipeline(TruncatedSVD(n_components=128, random_state=args.seed),
                            Normalizer(copy=False))
    Z = reducer.fit_transform(X)
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    tr_c = km.fit_predict(Z)
    te_c = km.predict(reducer.transform(vec.transform([text(ex) for ex in te_lab])))

    cmap = {}
    for ex, c in zip(tr_lab, tr_c):
        cmap[ph(ex["prompt"])] = [int(c)]
    for ex, c in zip(te_lab, te_c):
        cmap.setdefault(ph(ex["prompt"]), [int(c)])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cmap, f)
    print(f"wrote {len(cmap)} prompt->cluster entries -> {args.out}")

    print("cluster sizes:", sorted(Counter(int(c) for c in tr_c).values(), reverse=True))

    # Diagnostic only: how much do the discovered clusters resemble the PKU taxonomy?
    # Low purity means the clusters are a genuinely different partition rather than a
    # rediscovery of the annotated categories.
    byc = defaultdict(Counter)
    for ex, c in zip(tr_lab, tr_c):
        byc[int(c)][HARM_CATEGORIES[ex["category_indices"][0]]] += 1
    purity = sum(max(cc.values()) for cc in byc.values()) / len(tr_lab)
    print(f"cluster purity wrt PKU primary category = {purity:.3f} "
          f"(1.0 = clusters reproduce the taxonomy, ~0.1 = unrelated)")
    for c in sorted(byc)[:6]:
        print(f"  cluster {c:2d} (n={sum(byc[c].values()):4d}): {byc[c].most_common(2)}")


if __name__ == "__main__":
    main()
