import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("courses.csv")
df['text_features'] = df['course_title'].fillna('') + " " + df['tags'].fillna('')

# --- Streamlit UI ---
st.set_page_config(page_title="🎓 Course Recommender", layout="centered")
st.title(" Personalized Course Recommender")
st.markdown("Get course suggestions based on your **interests** and **career goals**.")

# === Input section ===
interest = st.text_input(" What topic or skill are you interested in?", "Python")
goal = st.text_input(" What is your desired field or career goal?", "Data Science")
top_n = st.slider(" How many course recommendations?", 5, 20, 10)

if st.button("✨ Show Recommendations"):
    # Combine user input
    user_query = interest + " " + goal

    # TF-IDF + Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text_features'])
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df['score'] = similarities
    recommendations = df.sort_values(by='score', ascending=False).head(top_n)

    if recommendations.empty:
        st.warning("No relevant courses found. Try different keywords.")
    else:
        st.success(f"Top {top_n} courses based on your interest in '{interest}' and goal '{goal}'")

        # Display recommendations
        for i, row in recommendations.iterrows():
            st.markdown(f"""
            ### {row['course_title']}
            -  **Platform**: {row['platform']}
            -  **Level**: {row['level']}
            -  **Rating**: {row['rating']}
            -  [Go to Course]({row['url']})
            ---
            """)

        # --- Charts ---
        st.markdown("###  Platform Trends")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=recommendations, x='platform', ax=ax1, palette='Set2')
        ax1.set_title("Top Platforms")
        st.pyplot(fig1)

        st.markdown("###  Difficulty Distribution")
        fig2, ax2 = plt.subplots()
        recommendations['level'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax2, colors=sns.color_palette('pastel'))
        ax2.set_ylabel("")
        ax2.set_title("Difficulty Breakdown")
        st.pyplot(fig2)

        st.markdown("###  Average Rating by Platform")
        fig3, ax3 = plt.subplots()
        avg_rating = recommendations.groupby("platform")["rating"].mean().sort_values(ascending=False)
        sns.barplot(x=avg_rating.index, y=avg_rating.values, ax=ax3, palette="Blues")
        ax3.set_ylabel("Avg Rating")
        ax3.set_title("Average Course Rating")
        st.pyplot(fig3)

