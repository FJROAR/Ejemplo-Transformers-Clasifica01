import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline
# Change `transformersbook` to your Hub username

model_id = "fjroar/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
custom_tweet = "Party at my house, tomorrow afternoon"
preds = classifier(custom_tweet, return_all_scores = True)
preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
