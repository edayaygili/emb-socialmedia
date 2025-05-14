import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Model yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Kategori açıklamaları
category_descriptions = {
    "Politics": "Includes news about government decisions, presidents or prime ministers, political parties, elections, debates in parliament, public policies, or political protests.",
    "Entertainment": "Covers topics about movies, TV shows, celebrities, Netflix releases, music albums, actors, singers, or entertainment awards like the Oscars.",
    "Technology": "Includes news or opinions about artificial intelligence (e.g., ChatGPT), new phone releases, apps, programming, tech companies, or internet trends.",
    "People": "Tweets about personal opinions, daily life, feelings, relationships, identity, experiences, or personal stories.",
    "Cryptocurrency": "Mentions Bitcoin, Ethereum, crypto trading, blockchain projects, NFTs, Web3, wallets, or cryptocurrency news.",
    "Social": "Topics related to society such as gender, identity, race, culture, social movements, human rights, or community stories.",
    "Sports": "Mentions football, basketball, tennis, sports teams, athletes like Ronaldo or Messi, match scores, tournaments, or the Olympics.",
    "Environment": "Includes discussions about climate change, pollution, sustainability, renewable energy, forest fires, or nature conservation.",
    "Business": "Covers companies, CEOs, business news, entrepreneurship, startups, layoffs, office work, or mergers and acquisitions.",
    "Science": "Includes scientific research, discoveries, experiments, physics, chemistry, biology, academic papers, or space exploration (e.g., NASA).",
    "Finance": "Talks about banks, loans, credit cards, savings, personal finance tips, taxes, or financial planning.",
    "Health": "Mentions hospitals, doctors, diseases, mental health, COVID-19, vaccines, nutrition, or fitness.",
    "Investing": "Includes tips or opinions on stocks, the stock market, real estate investment, crypto assets, or personal wealth growth.",
    "Economy": "Discusses inflation, unemployment, GDP, cost of living, economic growth, global markets, or economic policies.",
    "Law": "Mentions courts, lawsuits, legal cases, lawyers, criminal justice, police activity, constitutional rights, or new laws."
}

# Tahmin fonksiyonu
def predict_relevance(text, selected_category):
    selected_category_clean = selected_category.strip().title()
    if selected_category_clean not in category_descriptions:
        return "⚠️ Unknown category."
    
    text_vec = model.encode([text])[0]
    category_vec = model.encode([category_descriptions[selected_category_clean]])[0]
    similarity = cosine_similarity([text_vec], [category_vec])[0][0]
    label = "✅ Relevant" if similarity >= 0.12 else "❌ Non-Relevant"
    return label, similarity

# Streamlit arayüzü
st.title("📊 Content Relevance Checker")
st.markdown("Paste a tweet or short post and select a category to check if the content matches.")

user_input = st.text_area("Enter your text", placeholder="Paste your post or tweet...", height=200)
category = st.selectbox("Select Category", list(category_descriptions.keys()))

if st.button("Check Relevance"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        label, sim = predict_relevance(user_input, category)
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Similarity Score:** `{sim:.3f}`")
