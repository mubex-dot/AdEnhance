from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)


def process(text):
    # for j in range(len(text)):
    text2 = preprocess(text)
    encoded_input = tokenizer(text2, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    row = {
        "msg": text,
        "sentiment": config.id2label[ranking[0]],
        "value": scores[ranking[0]],
    }
    print(row)
    return row


textArr = [
    "Just got my hands on the 'EcoTech Solar-Powered Backpack,' and I'm absolutely loving it! 🎒🌞 Not only is it stylish and comfortable, but the built-in solar panels are a game-changer. They keep my devices charged on the go, and the eco-friendly design makes me feel good about reducing my carbon footprint. ♻️📱 Plus, it has tons of compartments and space for all my essentials. Highly recommend this innovative backpack! 👌 #EcoTechBackpack #SolarPower",
    "Just tried the 'GigaGlow SuperFlashlight' and I'm seriously unimpressed! 🙄 It claims to be the brightest flashlight ever, but it barely lit up my closet. 💡 Also, the battery died after just 10 minutes of use. 🤦‍♂️ Save your money and invest in something better! #ProductFail #GigaGlowDisappointment",
    "Just got my hands on the #NanoBlasterX, and it's a total letdown. 🙅‍♂️ The hype didn't match the reality. The build quality feels cheap, and it barely delivered half the promised performance. Save your money, folks! 💸👎 #Disappointed #NotWorthIt",
]

process(textArr[0])

# def process():
#     rows = []
#     for j in range(len(textArr)):
#         text2 = preprocess(textArr[j])
#         encoded_input = tokenizer(text2, return_tensors="pt")
#         output = model(**encoded_input)
#         scores = output[0][0].detach().numpy()
#         scores = softmax(scores)
#         ranking = np.argsort(scores)
#         ranking = ranking[::-1]
#         rows.append(
#             {
#                 "msg": textArr[j],
#                 "sentiment": config.id2label[ranking[0]],
#                 "value": scores[ranking[0]],
#             }
#         )
#     print(rows)


# process()
