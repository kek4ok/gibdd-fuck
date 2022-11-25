from aiogram import types, Dispatcher
from bot_setting import bot
from aiogram.dispatcher.filters import Text
from bot_tg.img_logic import get_random_image, rename_captcha
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext

class FSMForma(StatesGroup):
    img = State()

async def command_start(message: types.message, state: FSMContext):
    await FSMForma.img.set()

    global img
    img = get_random_image()
    async with state.proxy() as data:
        data['img'] = img
    p = open(img, 'rb')
    await FSMForma.next()
    await bot.send_photo(message.from_user.id, p) #Сюда надо фото где хуй

async def command_chech(message: types.message, state: FSMContext):
    await bot.send_message(message.from_user.id, "поймал")
    # Сюда функцию и передовай message.text
    async with state.proxy() as data:
        rename_captcha(message.text, data['img'])
    await state.finish()

def register_handlers_worker(dp: Dispatcher):
    dp.register_message_handler(command_start, Text(equals='1', ignore_case=True), state="*")
    dp.register_message_handler(command_chech)
