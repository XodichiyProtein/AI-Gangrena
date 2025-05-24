import io
import os
import json
import base64
from PIL import Image
import torch
from ollama import Client
from typing import Any

class AnalysisPhoto:
    def __init__(self):
        # Проверка доступности CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство: {self.device}")
        if not torch.cuda.is_available():
            print("Предупреждение: CUDA недоступен, модель будет работать на CPU.")

        # Инициализация клиента ollama
        try:
            self.client = Client(host='http://localhost:11434')
            # Проверка доступности сервера
            self.client.generate(model='llava', prompt="test", stream=False)
        except Exception as e:
            print(f"Ошибка подключения к серверу ollama: {str(e)}")
            raise

    def _image_to_base64(self, img: Image.Image) -> str:
        """Преобразование изображения в строку base64."""
        buffered = io.BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def ask_llava(self, tool_input: Any) -> str:
        """Обработка изображения и вопроса с использованием модели llava через ollama."""
        try:
            # Обработка входных данных
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    return "Ошибка: Неверный JSON-вход для инструмента AnalysisPhoto."

            if not isinstance(tool_input, dict) or "image_path" not in tool_input or "question" not in tool_input:
                return "Ошибка: Входные данные должны быть словарем с ключами 'image_path' и 'question'."

            image_path = os.path.normpath(tool_input["image_path"])
            question = tool_input["question"]

            if not os.path.exists(image_path):
                return "Ошибка: Файл изображения не найден."

            # Загрузка и преобразование изображения
            img = Image.open(image_path)
            base64_img = self._image_to_base64(img)

            # Выполнение запроса к модели llava через ollama
            response = self.client.generate(
                model='llava',
                prompt=question,
                images=[base64_img],
                stream=False,
                options={
                    "num_gpu": -1  # Автоматическое использование всех доступных GPU
                }
            )

            result = 'Описание от вопроса: ' + response['response']
            return result
        except Exception as e:
            return f"Ошибка: {str(e)}"

if __name__ == '__main__':
    bot = AnalysisPhoto()
    print(bot.ask_llava({'image_path' : r"C:\Users\kules\Downloads\photo_2025-05-12_22-24-43.jpg", 'question' : 'Who on the photo?'}))