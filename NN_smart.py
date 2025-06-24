import os
import io
import logging
from PIL import Image
import numpy as np
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_PATH = 'модельТормоз.keras'

try:
    model = load_model(MODEL_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь фото детали — я проверю её на дефекты."
    )

def predict_defect(img_path: str, threshold: float = 0.5) -> tuple:
    try:
        img = image.load_img(img_path, target_size=(300, 300), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array.reshape(1, 300, 300, 1)
        img_array = img_array / 255.0

        defect_prob = float(model.predict(img_array, verbose=0)[0][0])
        normal_prob = 1 - defect_prob
        prediction = 1 if defect_prob > threshold else 0

        return prediction, defect_prob, normal_prob
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка фото от пользователя"""
    try:
        logger.info(f"Получено фото от: {update.message.from_user.username}")
        await process_photo(update.message.photo[-1], update, context)
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке изображения.")

async def handle_channel_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка фото из канала"""
    try:
        logger.info("Получено фото из канала")
        await process_photo(update.channel_post.photo[-1], update, context, from_channel=True)
    except Exception as e:
        logger.error(f"Ошибка обработки фото из канала: {e}")

async def process_photo(photo, update, context, from_channel=False):
    """Общий обработчик фото"""
    photo_file = await photo.get_file()
    photo_bytes = io.BytesIO(await photo_file.download_as_bytearray())
    temp_path = 'temp_photo.jpg'
    with open(temp_path, 'wb') as f:
        f.write(photo_bytes.getbuffer())

    pred, defect_prob, normal_prob = predict_defect(temp_path)
    class_names = {0: "Дефект", 1: "Норма"}
    result_text = (
        f"🔍 Результат:\n"
        f"• Статус: <b>{class_names[pred]}</b>\n"
        f"• Норма: {normal_prob:.1%}\n"
        f"• Дефект: {defect_prob:.1%}"
    )

    if from_channel:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=result_text, parse_mode='HTML')
    else:
        await update.message.reply_text(result_text, parse_mode='HTML')

    os.remove(temp_path)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f'Update {update} вызвал ошибку: {context.error}')

def main() -> None:
    application = Application.builder().token("6774012378:AAE6PUiMZLsTs9VJW9JzKL-xztSO2WhK8qA").build() #7337132420:AAHkNg0FeFhQ2w5zorl3D8c_EKDbevZMNSk - esp
#6774012378:AAE6PUiMZLsTs9VJW9JzKL-xztSO2WhK8qA
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, handle_photo))
    application.add_handler(MessageHandler(filters.PHOTO & filters.ChatType.CHANNEL, handle_channel_photo))
    application.add_error_handler(error_handler)

    application.run_polling()
    logger.info("Бот запущен")

if __name__ == '__main__':
    main()
