import telebot
import test 
import torchvision
from PIL import Image
import io
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)

# Load the modified layers from your saved state_dict
saved_state_dict = torch.load("flower_classification_model.pth")
model_state_dict = model.state_dict()

# Update the model's state_dict with the saved state_dict for matching keys
for key in model_state_dict.keys():
    if key in saved_state_dict and "resnet_fc" in key:
        model_state_dict[key] = saved_state_dict[key]

model.load_state_dict(model_state_dict)


bot = telebot.TeleBot("6557253961:AAGVzm1yrU6hNf_YABjq7EJTmaT1lMwBRTE")


@bot.message_handler(content_types=["photo"])
def handler_image(message):
    file_info = bot.get.file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image = Image.open(io.BytesIO(downloaded_file))
    image = test.classify_image.predict_image(image, model)
    image = image.to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_class = output.max(1)

    class_names = ["тюльпан", "ромашка", "подсолнух", "роза", "одуванчик"]
    bot.reply_to(message, f"Это {class_names[predicted_class.item()]}")


bot.polling()
