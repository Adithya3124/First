import pandas as pd
from sklearn.model_selection import train_test_split as tts
import wittgenstein as lw
df = pd.read_csv("zoo.csv", names=[
"animal_name","hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize","class_type"])
df =df.drop(columns=["animal_name"])
df["class_type"] = (df["class_type"] == 1).astype(int)
train, test = tts(df, test_size=0.3, random_state=42)
model = lw.RIPPER().fit(train,class_feat="class_type")
print("Learned Rules:\n",model.ruleset_)
print("\nSample Prediction:\n",pd.Series(model.predict(test)).head())
