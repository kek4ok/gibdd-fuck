from aiogram.utils import executor
from bot_setting import dp


async def on_startup(_):
    print("bot work")


from headers import worker
worker.register_handlers_worker(dp)


executor.start_polling(dp, skip_updates=True, on_startup=on_startup)