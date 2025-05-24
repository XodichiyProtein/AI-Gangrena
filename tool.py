import json
from typing import Dict, Any

from googlesearch import search
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader, \
    UnstructuredWordDocumentLoader, UnstructuredRTFLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, \
    CSVLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader, UnstructuredMarkdownLoader, UnstructuredEmailLoader
from langchain.chains.conversation.base import ConversationChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms.ollama import Ollama
from diffusers import StableDiffusionXLPipeline
from ollama import Client
from PIL import Image
import base64
import io
import os
import torch
import re


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Code:
    def __init__(self, code_llm='coder'):
        self.code_llm = Ollama(model=code_llm, num_gpu=63)# Указываем слои для GPU)

    def ask_code(self, prompt) -> str:
        # question: str, file_context: str, search: str, file_text : str

        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                return "Error: Invalid JSON input for Code tool."

            # Validate required keys
        if not isinstance(prompt, dict) or not all(
                key in prompt for key in ["question", "file_context", "file_text"]):
            return "Error: Input must be a dictionary with 'question', 'file_context', 'file_text' keys."

        question : str= prompt["question"]
        file_context : str = prompt["file_context"]
        file_text : str= prompt["file_text"]

        cleaned = file_context.strip("[]").replace("'", "").replace('"', '')
        file_context = [file.strip() for file in cleaned.split(",")]


        f = CurrencyConverter()
        file_ctx = ''
        if file_context != None:
            for file in file_context:
                file_ctx += ('Название файла '+str(file) + ' : ' + ' Информация из файла '+str(f.load_file(file)))
        else:
            file_ctx = ''
        print("-"*50)
        conversation = ConversationChain(llm=self.code_llm)
        full_input = f"""
        ### Инструкции:
        Ты должен создать код по запросу пользователя. Выдавай код и дополнение к нему
        
        ### Контекст:
        - Код из файлов: {file_ctx}
        - {file_text}

        ### Запрос пользователя:
        {question}

        ### Ответ:
        """
        print(full_input)
        response = conversation.predict(input=full_input)
        return ('готовый код:' + response)


class GenerationPhoto:
    def __init__(self):
        pass

    def gen_img(self, prompt: str) -> str:
        try:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe = pipe.to("cuda")

            image = pipe(prompt).images[0]

            save_path = r"Z:\PyChatmProject\langchain\office_girl.png"  # <- Поменяйте на свой путь
            image.save(save_path)
            return 'Файл создан'
        except Exception as e:
            return e


class CurrencyConverter:
    def __init__(self):
        pass

    def load_file(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return "Error: File not found"

            file_path = file_path.strip().lower()

            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx") or file_path.endswith(".doc"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".rtf"):
                loader = UnstructuredRTFLoader(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                loader = UnstructuredExcelLoader(file_path)
            elif file_path.endswith(".pptx") or file_path.endswith(".ppt"):
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file_path.endswith(".html") or file_path.endswith(".htm"):
                loader = UnstructuredHTMLLoader(file_path)
            elif file_path.endswith(".epub"):
                loader = UnstructuredEPubLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_path.endswith(".eml") or file_path.endswith(".msg"):
                loader = UnstructuredEmailLoader(file_path)
            else:
                # Для всех остальных форматов используем универсальный загрузчик
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            return f"Error processing file: {str(e)}"


class AnalysisPhoto:
    def __init__(self):
        # Проверка доступности CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            result = f'Описание от вопроса: {response['response']}'
            return result
        except Exception as e:
            return f"Ошибка: {str(e)}"

class Enternet:
    def __init__(self):
        pass

    def google_search(self, query):
        """
        Выполняет поиск в Google с использованием библиотеки googlesearch.
        """
        try:
            results = []
            for url in search(query, num_results=10, lang="ru"):
                results.append({"url": url})
            return results
        except Exception as e:
            return f"Ошибка при выполнении поиска: {str(e)}"



if __name__ == '__main__':
    bot = Code()
    print(bot.ask_code(
        {"question": "Что можно улучшить в коде из файлов? Выдай готовый код", "file_context":"['Z:\\PyChatmProject\\langchain\\v2\\tool.py', 'Z:\\PyChatmProject\\langchain\\v2\\main.py']", "file_text":""}))
