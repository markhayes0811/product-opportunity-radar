# Product Opportunity Radar — Mark Hayes

A demo that ranks product opportunities using unmet demand (search gaps), price-sensitivity, review pain points, and competitor gaps. Includes lifecycle-aligned GitHub Issue templates.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/opportunity_pipeline.py
streamlit run app.py
```

## Colab
Use `Product_Opportunity_Radar_Colab.ipynb` to run end-to-end in Colab, visualize results, and optionally launch Streamlit via a public URL (pyngrok).

## LinkedIn blurb
> Shipped **Product Opportunity Radar**—turns search, sales, reviews, and competitor signals into a ranked list of product bets. Includes a Streamlit dashboard + lifecycle issue templates. Dummy data included.
