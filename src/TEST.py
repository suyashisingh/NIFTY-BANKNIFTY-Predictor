import os

# List everything under src
print("src:", os.listdir(r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src"))

# List everything under src\data
print("src\\data:", os.listdir(r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data"))

# List everything under src\data\processed
print("src\\data\\processed:", os.listdir(r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed"))


# import pandas as pd
#
# features_path = r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/nifty_engineered_features2.csv"
# df = pd.read_csv(features_path)
# print(df.columns.tolist())



# import pandas as pd
# from prophet import Prophet
# import joblib
#
# # Load your BANKNIFTY data (ensure columns: 'Date', 'Close')
# df = pd.read_csv(r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/data/processed/banknifty_merged_preprocessed.csv")
# df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
# df['ds'] = pd.to_datetime(df['ds'])
#
# model = Prophet()
# model.fit(df)
# joblib.dump(model, r"D:/intern-25/NIFTY_BANKNIFTYPredictor/pythonProject/src/models/prophet_next_month_banknifty.pkl")
