from fastapi import FastAPI
from sentiment_analysis import process
import requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

textArr = [
    "Just got my hands on the 'EcoTech Solar-Powered Backpack,' and I'm absolutely loving it! 🎒🌞 Not only is it stylish and comfortable, but the built-in solar panels are a game-changer. They keep my devices charged on the go, and the eco-friendly design makes me feel good about reducing my carbon footprint. ♻️📱 Plus, it has tons of compartments and space for all my essentials. Highly recommend this innovative backpack! 👌 #EcoTechBackpack #SolarPower",
    "Just tried the 'GigaGlow SuperFlashlight' and I'm seriously unimpressed! 🙄 It claims to be the brightest flashlight ever, but it barely lit up my closet. 💡 Also, the battery died after just 10 minutes of use. 🤦‍♂️ Save your money and invest in something better! #ProductFail #GigaGlowDisappointment",
    "Just got my hands on the #NanoBlasterX, and it's a total letdown. 🙅‍♂️ The hype didn't match the reality. The build quality feels cheap, and it barely delivered half the promised performance. Save your money, folks! 💸👎 #Disappointed #NotWorthIt",
]


@app.get("/")
async def root():
    res = requests.get("https://dummyjson.com/products/1")
    data = res.json()
    return {"message": data}


@app.get("/comment/{post_id}/{comment}")
async def make_comment(post_id: int, comment: str):
    result = process(textArr[0])
    result = jsonable_encoder(result)
    return JSONResponse(content=result)


@app.get("/like/{post_id}/")
async def make_comment():
    return {"message": "like"}
