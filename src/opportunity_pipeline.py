
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load():
    return (
        pd.read_csv(DATA_DIR/"transactions.csv"),
        pd.read_csv(DATA_DIR/"search_logs.csv"),
        pd.read_csv(DATA_DIR/"reviews.csv"),
        pd.read_csv(DATA_DIR/"catalog.csv"),
        pd.read_csv(DATA_DIR/"competitors.csv"),
    )

def unmet_demand(searches):
    s = searches.copy()
    s["unmet"] = ((s["results_found"]==0) | (s["clicks"]==0)).astype(int)
    g = s.groupby("query").agg(searches=("query","count"), unmet=("unmet","sum"), add_to_cart=("added_to_cart","sum")).reset_index()
    g["unmet_rate"] = g["unmet"]/g["searches"]
    g["conversion_rate"] = (g["add_to_cart"]/g["searches"]).fillna(0)
    g["unmet_signal"] = g["unmet_rate"]*(1-g["conversion_rate"])
    return g.sort_values("unmet_signal", ascending=False)

def price_sens(sales):
    out=[]
    for pid,g in sales.groupby("product_id"):
        corr = 0.0
        if g["discount_pct"].nunique()>1 and g["units"].nunique()>1:
            corr = np.corrcoef(g["discount_pct"], g["units"])[0,1]
        out.append({"product_id":pid,"price_sensitivity":corr})
    return pd.DataFrame(out)

def pain_points(reviews, top_k=8):
    low = reviews[reviews["rating"]<=3].copy()
    if low.empty: return pd.DataFrame(columns=["product_id","pain_points"])
    docs = low.groupby("product_id")["review_text"].apply(" ".join).reset_index()
    vect = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1,2))
    try:
        X = vect.fit_transform(docs["review_text"])
    except ValueError:
        return pd.DataFrame(columns=["product_id","pain_points"])
    terms = np.array(vect.get_feature_names_out())
    rows=[]
    for i,pid in enumerate(docs["product_id"]):
        row=X[i].toarray().ravel(); idx=row.argsort()[::-1][:top_k]
        rows.append({"product_id":pid,"pain_points":", ".join(terms[idx])})
    return pd.DataFrame(rows)

def comp_gaps(catalog, competitors):
    cat = catalog.copy(); cat["our_feats"]=cat["features"].str.lower().fillna("")
    comp = competitors.copy(); comp["comp_feats"]=comp["key_features"].str.lower().fillna("")
    rows=[]
    for c in sorted(set(cat["category"]) & set(comp["category"])):
        ours = cat[cat["category"]==c]; comps=comp[comp["category"]==c]
        our_tokens=set(", ".join(ours["our_feats"]).replace("/"," ").split(", "))
        comp_tokens=set(", ".join(comps["comp_feats"]).replace("/"," ").split(", "))
        gap=[t.strip() for t in comp_tokens-our_tokens if t.strip()]
        rows.append({"category":c, "missing_features":", ".join(sorted(set(gap)))[:200]})
    return pd.DataFrame(rows)

def compose(unmet, price, pain, catalog, gaps):
    kw={"watch":"Wearables","vacuum":"Home","air fryer":"Kitchen","lantern":"Outdoors","treadmill":"Fitness","cat":"Pets","pet":"Pets","ice maker":"Kitchen"}
    def map_cat(q):
        ql=q.lower()
        for k,v in kw.items():
            if k in ql: return v
        return "Unknown"
    unmet=unmet.copy(); unmet["category"]=unmet["query"].apply(map_cat)
    unmet_cat=unmet.groupby("category").agg(unmet_signal=("unmet_signal","sum"), search_volume=("searches","sum")).reset_index()
    price2=price.merge(catalog[["product_id","category","name"]], on="product_id", how="left")
    price_cat=price2.groupby("category")["price_sensitivity"].mean().reset_index()
    pain2=pain.merge(catalog[["product_id","category","name"]], on="product_id", how="left")
    pain_cat=pain2.groupby("category")["pain_points"].apply(lambda s: ", ".join(s.dropna().tolist())[:200]).reset_index()
    opp=(unmet_cat.merge(price_cat, on="category", how="left").merge(pain_cat, on="category", how="left").merge(gaps, on="category", how="left"))
    for col in ["unmet_signal","price_sensitivity"]:
        vals=opp[col].fillna(0)
        opp[col+"_norm"]=(vals-vals.min())/(vals.max()-vals.min()) if vals.max()>vals.min() else 0.0
    opp["opportunity_score"]=0.55*opp["unmet_signal_norm"]+0.25*opp["price_sensitivity_norm"]+0.20*(opp["missing_features"].notna().astype(int))
    def rec(row):
        ideas=[]
        if isinstance(row.get("missing_features",""),str) and row["missing_features"]: ideas.append(f"Add: {row['missing_features']}")
        if isinstance(row.get("pain_points",""),str) and row["pain_points"]: ideas.append(f"Address: {row['pain_points']}")
        if row.get("price_sensitivity",0)>0.1: ideas.append("Test price promotions or bundles")
        return "; ".join(ideas)[:300]
    opp["recommended_actions"]=opp.apply(rec, axis=1)
    cols=["category","search_volume","unmet_signal","price_sensitivity","missing_features","pain_points","opportunity_score","recommended_actions"]
    return opp.sort_values("opportunity_score", ascending=False)[cols]

def run():
    sales, searches, reviews, catalog, competitors = load()
    opp = compose(unmet_demand(searches), price_sens(sales), pain_points(reviews), catalog, comp_gaps(catalog, competitors))
    out = DATA_DIR/"opportunities.csv"; opp.to_csv(out, index=False); print("Saved", out)

if __name__ == "__main__":
    run()
