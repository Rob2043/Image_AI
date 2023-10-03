import telebot as tlb
import os
import torch
import torchvision
from PIL import Image

bot = tlb.TeleBot("6452093352:AAHX_OrZWAM4sUq7YBUS5yoF-kzjlqe1BCI")


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "хей жду фото:")


@bot.message_handler(content_types=["photo"])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        # Сохраняем информацию о фотографии
        file_info = bot.get_file(message.photo[-1].file_id)

        # Загружаем фотографию
        downloaded_file = bot.download_file(file_info.file_path)

        # Создаем новый файл и сохраняем туда фотографию
        with open("image.jpg", "wb") as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Фотография успешно сохранена!")

        # Загружаем изображение с помощью PIL
        img = Image.open("image.jpg")

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (224, 224)
                ),  # приводим картинки к одному размеру
                torchvision.transforms.ToTensor(),  # упаковывем их в тензор
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],  # нормализуем картинки по каналам
                ),
            ]
        )

        # Применяем трансформации к изображению
        img_t = transform(img)
        img_t = img_t.unsqueeze(0)  # добавляем размерность батча

        model = torch.load("C:\Telegram bot\Image_AI\\AIF_3.0.pth")
        model.eval()

        with torch.no_grad():
            prediction = model(img_t)

            # Получаем индекс наиболее вероятного класса
            _, predicted_idx = torch.max(prediction, 1)

            # Предполагая, что у вас есть список классов цветов
            classes = [
                "daisy",
                "dandelion",
                "rose",
                "sunflower",
                "tulip",
            ]  # Замените на свой список классов

            # Получаем наиболее вероятный класс
            predicted_class = classes[predicted_idx]

            bot.reply_to(message, f"Я думаю, это {predicted_class}!")

    except Exception as e:
        bot.reply_to(message, e)


bot.polling(True)
