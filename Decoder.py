import pandas as pd

# Replace 'your_file.json' with the path to your JSON file
df = pd.read_json(r"C:\Users\ASUS\Downloads\yelp_academic_dataset_review.json")

# To view the first few rows of the dataframe
print(df.head())


# import torch
# from torchtext.models import RobertaClassificationHead, LSTMClassifier
# from torchtext.functional import to_tensor

# # Load pretrained LSTM sentiment model
# from torchtext.models import LSTMClassificationHead, LSTMEncoder

# # Example: Pretrained sentiment model on IMDb
# from torchtext.models import RNNModel

# # Load pretrained RNN model
# bundle = torchtext.models.RNNModel.from_pretrained("lstm-imdb")

# # Move to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = bundle.to(device)

# # Evaluate sentiment of a sentence
# text = "This movie was surprisingly good!"
# input_batch = bundle.transform(text)

# with torch.no_grad():
#     prediction = model(input_batch)
#     label = torch.argmax(prediction, dim=1).item()

# print("Positive" if label == 1 else "Negative")