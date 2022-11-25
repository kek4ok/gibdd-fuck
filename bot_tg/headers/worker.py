from aiogram import types, Dispatcher
from bot_setting import bot
from aiogram.dispatcher.filters import Text
from bot_tg.img_logic import get_random_image, rename_captcha


async def command_start(message: types.message):
    global img
    img = get_random_image()
    p = open(img, 'rb')
    await bot.send_photo(message.from_user.id, p) #Сюда надо фото где хуй

async def command_chech(message: types.message):
    await bot.send_message(message.from_user.id, "поймал")
    # Сюда функцию и передовай message.text
    rename_captcha(message.text, img)

def register_handlers_worker(dp: Dispatcher):
    dp.register_message_handler(command_start, Text(equals='1', ignore_case=True), state="*")
    dp.register_message_handler(command_chech)
