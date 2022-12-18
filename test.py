from data import load_model,predict_sentence

model = load_model()

print(predict_sentence(model, input("Enter the sentence: ")))
