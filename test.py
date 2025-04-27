import joblib
from sklearn.exceptions import NotFittedError

def load_model_and_vectorizer(model_path, vectorizer_path):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

def test_model(model, vectorizer, input_text):
    try:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)
        return prediction
    except NotFittedError as e:
        print(f"Error: The model or vectorizer is not fitted. {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        exit(1)

if __name__ == "__main__":
    model_path = "svm_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    while True:
        input_text = input("Enter text to test the model (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting the program.")
            break
        prediction = test_model(model, vectorizer, input_text)
        print(f"Model Prediction: {prediction[0]}")


# Few prompts to test the model:
# AI:
# 1. Maintaining a healthy work-life balance is crucial for fostering long-term productivity and personal well-being in today's fast-paced environment.
# 2. The rapid advancements in artificial intelligence are reshaping industries and creating new opportunities for innovation and efficiency.
# 3. Embracing a growth mindset can significantly enhance one's ability to adapt to challenges and learn from experiences, ultimately leading to personal and professional development.
