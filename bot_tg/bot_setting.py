from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

storage = MemoryStorage()

bot = Bot(token='5986363212:AAGEJMq1qK5wCQfY4lXr8eSMMDVH9eaJUqk')
dp = Dispatcher(bot, storage=storage)