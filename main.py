import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

from tool import *


class AI_Bot_v2:
    def __init__(self):
        self.base_path = Path('.')
        self.global_wish_path = 'admin_wish.json'

        self.converter_tool = Tool(
            name="Анализ файла",
            func=CurrencyConverter().load_file,
            description='''Читает файл и возвращает текст из него.
                         Формат: просто укажите полный путь к файлу
                         Пример: "C:/Users/kules/Downloads/AyuGram Desktop/8.DOCX"'''
        )

        self.search_tool = Tool(
            name="Поиск Google",
            func=Enternet.google_search,
            description="""Поиск Google. Используй для получения актуальной информации из интернета.
                Формат: укажите вопрос или ключевое слово для поиска.
                Пример: 'Мультфильмы Disney'"""
        )

        self.AnalPhoto = Tool(
            name="Анализировать фото",
            func=AnalysisPhoto().ask_llava,
            description=(
                '''Анализирует фото. Задавать вопросы на англисском языке. Требует строгий JSON-формат:
                "{"image_path": "полный_путь.jpg", "question": "ваш_вопрос"}"
                Пример:
                "{"image_path": "C:/Users/kules/photo.jpg", "question": "Что на фото?"}"'''
            )
        )

        self.CodeGen = Tool(
            name='Работа с кодом',
            func=Code().ask_code,
            description='''Нейронная сеть которая работает с кодом, может редактировать, создавать, исправлять.
            Формат ввода строгий json: "{"question": "Вопрос пользователя", "file_context":"список файлов через запятую", "file_text":"данные из файлов" "
            Пример: {"question": "Вопрос пользователя", "file_context":"['C:\\Users\\kules\\Downloads\\AyuGram Desktop\\20230620_194957.jpg']", "file_text":"данные из файлов" " '''
        )
        self.PhotoGen = Tool(
            name='ГЕНИРАЦИЯ ФОТО',
            func=GenerationPhoto().gen_img,
            description='''Создаёт изображение по тексту. Используйте в запросе английский язык для более качественного ответа. Если пользователь просит сделать изображение то использовать этот интсрумент. Формат: 'вопрос пользователя' '''
        )

    # Storage---------------------------------------------------------------------------------------------------------->

    def _load_storage(self, ChatID: str, UserID: str) -> ConversationBufferMemory:
        chat_memory = FileChatMessageHistory(f"{UserID}\\{ChatID}.json")
        memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)
        return memory

    def _save_storage(self, UserID: str, ChatID: str, Human_Message: str, AI_Message: str = '') -> None:
        chat_memory = FileChatMessageHistory(f"{UserID}\\{ChatID}.json")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        chat_memory.add_user_message(f"[{timestamp}] {Human_Message}")
        if AI_Message:
            chat_memory.add_ai_message(f"[{timestamp}] {AI_Message}")
        print("Save")

    # Wish User and Admin---------------------------------------------------------------------------------------------->
    def _ensure_user_wish_file(self, user_id: str) -> Path:
        user_dir = self.base_path / user_id
        user_dir.mkdir(exist_ok=True)
        wish_file = user_dir / "UserWish.json"

        if not wish_file.exists():
            with wish_file.open('w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)

        return wish_file

    def add_wish(self, user_id: str = '0', human_message: str = '', ai_message: str = '') -> None:
        try:
            wish_file = self._ensure_user_wish_file(user_id)

            # Чтение существующих пожеланий
            with wish_file.open('r', encoding='utf-8') as f:
                data: List[str] = json.load(f)

            # Добавление нового пожелания
            data.append(f"I: {human_message} AI: {ai_message}")

            # Сохранение обновлённых пожеланий
            with wish_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Ошибка при добавлении пожелания для пользователя {user_id}: {e}")

    def _load_wish(self, user_id: str) -> str:
        try:
            wish_file = self._ensure_user_wish_file(user_id)

            with wish_file.open('r', encoding='utf-8') as f:
                data: List[str] = json.load(f)

            if not data:
                return "Пожелания пользователя отсутствуют."

            return f"Советы по общению (учитывай, но не упоминай): {data}"

        except Exception as e:
            print(f"Ошибка при загрузке пожеланий для пользователя {user_id}: {e}")
            return "Ошибка при загрузке пожеланий."

    def add_admin_wish(self, wish_message: str) -> None:
        try:
            # Читаем существующие пожелания
            with open(self.global_wish_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Добавляем новое пожелание
            data.append(wish_message)

            # Сохраняем обновлённые пожелания
            with open(self.global_wish_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Ошибка при добавлении глобального пожелания: {e}")

    def _load_admin_wishes(self) -> str:
        try:
            # Создаём файл, если он не существует
            if not os.path.exists(self.global_wish_path):
                with open(self.global_wish_path, 'w', encoding='utf-8') as file:
                    json.dump([], file, ensure_ascii=False, indent=4)

            with open(self.global_wish_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not data:
                    return "Глобальные пожелания отсутствуют."
                return f"Глобальные советы по общению (учитывай, но не упоминай): {data}"
        except Exception as e:
            print(f"Ошибка при загрузке глобальных пожеланий: {e}")
            return "Ошибка при загрузке глобальных пожеланий."

    # Base fun--------------------------------------
    def ask(self, prompt: str, UserID: str = '0', ChatID: str = '0', file_path: list = []) -> str:
        user_prompt = {'output': f'Вопрос: {prompt}, Используемые файлы: {str(file_path)}'}
        result = {'output': ''}
        try:
            self._load_wish(UserID)
            # Инициализация модели и агента
            model = OllamaLLM(
                model='gemma3:12b',
                temperature=0.1,
                top_p=0.7,
                frequency_penalty=0.3,
                timeout=60
            )
            tools = [self.converter_tool, self.CodeGen, self.PhotoGen, self.AnalPhoto, self.search_tool]
            react_prompt = PromptTemplate.from_template(
                """You are an intelligent AI assistant with access to the following tools:

                {tools}

                ### Instructions:
                - Analyze the user's question and select the most appropriate tool to answer it.
                - If no tool is needed, provide the answer directly.
                - Do not repeat actions unnecessarily. Stop after finding a relevant answer or if no tool is applicable.
                - If a tool returns an error or unclear result, report it and proceed to a final answer.
                - Avoid looping: you have a maximum of 2 tool calls before providing a final answer.

                """+f"""### User Wishes:
                Mandatory wishes: {self._load_admin_wishes()}
                User wishes: {self._load_wish(UserID)}
                Available files: {file_path}

                ### Dialogue History:
                {self._load_storage(ChatID, UserID).load_memory_variables({})["history"]}"""+"""

                ### Answer Format:
                Question: {input}
                Thought: Reason about which tool to use or if a direct answer is possible
                Action: Selected tool (one of [{tool_names}]) or "None" if no tool is needed
                Action Input: Input data for the tool (if applicable)
                Observation: Result of tool execution (if applicable)
                Thought: I have a final answer
                Final Answer: Final response to the user

                ### Let's get started!
                Question: {input}
                Thought: {agent_scratchpad}"""
            )
            agent = create_react_agent(
                model,
                tools,
                react_prompt
            )
            full_input = f"""
                    ### Инструкции:
                    Обязательные пожелания для работы: {self._load_admin_wishes()}
                    Пожелания пользователя для работы: {self._load_wish(UserID)}
                    Доступные файлы: {file_path}
                    
                    ### История:
                    История диалога с пользователем: {self._load_storage(ChatID, UserID).load_memory_variables({})}                    
                    ### Запрос пользователя:
                    {prompt}

                    ### Ответ:
                    """

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate",
                return_intermediate_steps=True,
                max_execution_time=300,
                tool_exception_handler=lambda e: f"Ошибка инструмента: {str(e)}"
            )
            # Пример использования
            result = agent_executor.invoke({
                "input": full_input
            })
            user_prompt = f'Вопрос: {prompt}, Используемые файлы: {str(file_path)}'

        except Exception as e:
            return str(e)

        self._save_storage(UserID=UserID, ChatID=ChatID, Human_Message=user_prompt, AI_Message=result['output'])
        return result['output']


'''Да она меня тупо в игнор кинула, 1 раз на улице встретились да и всё. По итогу 2 дня подряд меж ляшек у неё повалялся
И всё
Важный текст для релиза!'''

if __name__ == '__main__':
    bot = AI_Bot_v2()
    print(bot._load_storage(ChatID='6', UserID='0').load_memory_variables({})["history"])
    # print(bot.ask(ChatID='4', prompt='Что на этих фото', file_path=[r'Z:\Screenshots\бебра.png', r'Z:\Screenshots\Снимок экрана 2025-04-21 223750.png']))
    print(bot.ask(ChatID='6', prompt='Что можно улучшить в коде из файлов? Выдай готовый код',
                  file_path=[r'Z:\PyChatmProject\langchain\v2\tool.py', r'Z:\PyChatmProject\langchain\v2\main.py']))
