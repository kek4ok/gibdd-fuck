from aiogram import types, Dispatcher
from bot_setting import bot
from aiogram.dispatcher.filters import Text


async def command_start(message: types.message):
    await bot.send_message(message.from_user.id, "хуй" ) #Сюда надо фото где хуй

async def command_chech(message: types.message):
    await bot.send_message(message.from_user.id, "поймал" )
    # Сюда функцию и передовай message.text

def register_handlers_worker(dp: Dispatcher):
    dp.register_message_handler(command_start, Text(equals='1', ignore_case=True), state="*")
    dp.register_message_handler(command_chech)
