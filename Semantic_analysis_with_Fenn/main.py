# Fenn is a lightweight framework for building and deploying machine learning models. It provides a simple interface for defining, 
# training, and deploying models, as well as managing experiments and tracking metrics. Fenn is designed to be easy to use and flexible, 
# allowing users to focus on building models rather than managing infrastructure. honestly its the best thing that i have used in 
# recent times. It is a great tool for anyone who wants to build and deploy machine learning models quickly and easily.
# curently I am just trying it out for simpler projects


print("DEBUG: Starting the sentiment analysis model training script...")
from fenn import Fenn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

app = Fenn()
app.disable_disclaimer()

REVIEWS = [
    ("I love this product! It works great and exceeded my expectations.", 1),
    ("This is the worst purchase I've ever made. Completely disappointed.", 0),
    ("The quality is decent for the price, but not outstanding.", 1),
    ("I would not recommend this to anyone. Very poor performance.", 0),
    ("Absolutely fantastic! I will definitely buy this again.", 1),
    ("Terrible experience. The product broke after a week of use.", 0),
    ("It's okay, but I've seen better options available.", 1),
    ("I regret buying this. It didn't meet my needs at all.", 0),
    ("Great value for money. I'm very satisfied with my purchase.", 1),
    ("Not worth the hype. I expected more from this product.", 0)]

@app.entrypoint
def main(args):
    print("--- Sentiment Analysis Model Training ---")

    # Extract settings from Fenn.yaml via args
    test_size = args['data']['test_size']
    random_state = args['data']['random_state']
    max_features = args['train']['vectorizer']['max_features']

    # seprate texts and labels
    texts, labels = zip(*REVIEWS)

    # train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state)

    #feature extraction and model training
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    print("Training model...")
    model = LogisticRegression()
    model.fit(X_train, train_labels)

    # evaluate the model
    predictions = model.predict(x_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nModel Accuracy on test set: {accuracy * 100:.1f}%")

    #real_time prediction test
    sample_texts = [
        "this product is amazing! I am very happy with my purchase.",
        "I am extremely disappointed with this product. It broke after one use.",
    ]

    print("--- Running real test predictions ---")
    X_sample = vectorizer.transform(sample_texts)
    sample_predictions = model.predict(X_sample)

    for text, predictions in zip(sample_texts, sample_predictions):
        sentiment = "Positive" if predictions == 1 else "Negative"
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

    print("--- sentiment pipeline execution completed ---")


if __name__ == "__main__":
    app.run()