def find_s(dataset):
    H = dataset[0][:-1]       
    for ex in dataset:
        if ex[-1] == "Yes":           
            for i in range(len(H)):
                if H[i] != ex[i]:
                    H[i] = "?"
    return H
dataset = [ ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'], ['Sunny','Warm','High','Strong','Warm','Same','Yes'], ['Rainy','Cold','High','Strong','Warm','Change','No'], ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]
print("Most Specific Hypothesis:", find_s(dataset))
