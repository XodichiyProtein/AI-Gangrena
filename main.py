import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import PromptTemplate

from TechTool import ImageAnalysisInput, AnalysisPhoto
from Tools import AITools

import os
import json

from langchain_ollama import OllamaLLM


class AI_Bot_v2:
    def __init__(self):
        self.base_path = Path('.')
        self.global_wish_path = 'admin_wish.json'
        self.tools = AITools().get_all_tools()
        self.tools_name = AITools().get_all_tools_name()

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
            # Load user and admin wishes
            user_wishes = self._load_wish(UserID)
            admin_wishes = self._load_admin_wishes()

            # Initialize model
            model = OllamaLLM(
                model='mixtral:8x7b',
                temperature=0.1,
                tfs_z=0.95,  # Tail free sampling
                typical_p=0.95,  # Typical sampling
                repeat_penalty=1.1,  # Penalty за повторения
                top_p=0.7,
                frequency_penalty=0.3,
                timeout=60,
                gpu_layers=40  # Количество слоёв для выгрузки на GPU (максимально возможное)
            )

            history = self._load_storage(ChatID, UserID)
            react_prompt = PromptTemplate.from_template(
                """You are an AI assistant who processes requests: from writing code to finding information. Available tools:
    
                {tools}
    
                ### Instructions:
                - Read the request and decide if a tool is needed or if you can respond directly.
                - For code, use 'Work with code'. For photos, use 'Analyze photos' or 'Generate photos'.
                - If the request is simple or does not require tools, specify Action: None and give an answer.
                - If the tool returned an error or the result is sufficient, finish with Final Answer. Do not repeat actions unless necessary.
                - Maximum 1 tool call for simple requests to avoid loops.
                - Consider chat history, admin and user wishes, files if specified.
                - Always respond in the format: Thought, Action, Action Input, Observation, Final Answer.
                """ + f"""
                ### Context:
                Admin Wishes: {admin_wishes}
                User Wishes: {user_wishes}
                Available Files: {file_path}
                """ + """
                ### Response Format:
                Thought: [Request Analysis and Action Selection]
                Action: [One of {tool_names} or "None"]
                Action Input: [Tool Data or Empty]
                Observation: [Tool Result or Empty]
                Final Answer: [User Response]
    
                """ + """### Query:
                {input}
                Thought: {agent_scratchpad}
                """
            )

            # Create agent
            agent = create_react_agent(
                model,
                self.tools,
                react_prompt
            )

            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                memory=history,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,  # Fixed typo: Fasle -> True
                max_iterations=10,
                max_execution_time=300,
                tool_exception_handler=lambda e: f"Ошибка инструмента: {str(e)}"
            )

            # Выполняем запрос агента
            result = agent_executor.invoke({
                "input": prompt
            })


        except Exception as e:
            print(str(e))

        self._save_storage(UserID=UserID, ChatID=ChatID, Human_Message=user_prompt['output'],
                           AI_Message=result['output'])
        del model
        return result['output']


'''Да она меня тупо в игнор кинула, 1 раз на улице встретились да и всё. По итогу 2 дня подряд меж ляшек у неё повалялся
И всё

Я опять кадрю альтуху которую уже код-рил(ил)

Да нихуя, она к другу бывшего ушла. Я у неё позавчера был дома, и была одета как шлюха

Да я пока этот код писал успел сменить 2 альтухи! 01.06.2025

Важный текст для релиза!'''

if __name__ == '__main__':
    bot = AI_Bot_v2()
    result = bot.ask(
        ChatID='10',
        prompt='"Z:/PyChatmProject/langchain/v2/hui.txt" что на этом файле?',
    )
    print(result)
