"""
This project was created by Dennis Rotnov. Feel free to use, reuse and modify it, but please retain this authorship attribution.

"""

import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import math
from more_itertools import batched
from better_profanity import profanity
from typing import Iterator, Union, Optional

load_dotenv()


class OpenAIAPI():

    def text_cleaner(self, prompt: str) -> str:
        """
        Cleans the text by removing stopwords to make prompt shorter and censoring profane words.

        Args:
            prompt (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        english_stopwords = stopwords.words('english')
        tokenized_words = nltk.tokenize.word_tokenize(prompt)
        if len(tokenized_words) > 30:
            prompt = " ".join([word for word in tokenized_words if word not in english_stopwords])
            # prompt = (" ").join(prompt)
            return profanity.censor(prompt)
        return profanity.censor(prompt)

    def batched_prompt(self, prompt: str, token_size: int) -> Iterator[str]:
        """
        Splits a long prompt into batches of certain length expressed in tokens.

        Args:
            prompt (str): The prompt to be split.
            token_size (int): The size of a text batch in tokens.

        Returns:
            Iterator[str]: An iterator of batches of text.
        """
        prompt = self.text_cleaner(prompt)
        total_tokens = math.ceil((len(prompt)/4))
        batches = math.ceil(total_tokens / token_size)
        if total_tokens > token_size:
            for bacth in batched(prompt, math.ceil(len(prompt)/batches)):
                yield " ".join(bacth)
        yield prompt

    async def generate_chat_response(self, prompt: str, *, model: str, api_key: Optional[str]=os.getenv('OPENAI_API_KEY')) -> str:
        """
        Generate a response from the OpenAI API based on the chat model.

        Args:
            prompt (str): The prompt to be used for generating the response.
            model (str): The model to be used for generating the response.
            api_key (Optional[str]): The API key to be used for generating the response. If not provided, the value of the OPENAI_API_KEY environment variable will be used.

        Returns:
            str: The generated response.
        """
        prompt = self.text_cleaner(prompt)
        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=model,
            # max_tokens=4000 - (math.ceil(len(prompt)/4)),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    async def generate_completetion_response(self, prompt: str, *, model: str, api_key: Optional[str]=os.getenv('OPENAI_API_KEY')) -> str:
        """
        Generate a completion response from the OpenAI API based on the completion model.

        Args:
            prompt (str): The prompt to be used for generating the response.
            model (str): The model to be used for generating the response.
            api_key (Optional[str]): The API key to be used for generating the response. If not provided, the value of the OPENAI_API_KEY environment variable will be used.

        Returns:
            str: The generated response.
        """
        prompt = self.text_cleaner(prompt)
        client = AsyncOpenAI(api_key=api_key)
        completions = await client.completions.create(
            model=model,
            prompt=prompt,
            # max_tokens=4000 - (math.ceil(len(prompt)/4)),
            n=1,
            top_p=1,
            stop=None,
            temperature=0.5,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        return completions.choices[0].text

    async def generate_image(self, prompt: str, *, model: str, api_key: Optional[str]=os.getenv('OPENAI_API_KEY')) -> str:
        """
        Generate an image from the OpenAI API based on the image model.

        Args:
            prompt (str): The prompt to be used for generating the image.
            model (str): The model to be used for generating the image.
            api_key (Optional[str]): The API key to be used for generating the image. If not provided, the value of the OPENAI_API_KEY environment variable will be used.

        Returns:
            str: The generated image URL.
        """
        client = AsyncOpenAI(api_key=api_key)
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url

    async def generate_batches(self, prompt: str, *, method: str, model: str, api_key: Optional[str]=os.getenv('OPENAI_API_KEY'), token_size: int) -> str:
        """
        Generate multiple responses in parallel using the OpenAI API.

        Args:
            prompt (str): The prompt to be used for generating the responses.
            method (str): The method to be used for generating the responses. Can be either "chat" or "completions".
            model (str): The model to be used for generating the responses.
            api_key (Optional[str]): The API key to be used for generating the responses. If not provided, the value of the OPENAI_API_KEY environment variable will be used.
            token_size (int): The size of a text batch in tokens.

        Returns:
            str: A string containing the responses generated by the API.
        """
        batches = self.batched_prompt(prompt, token_size)
        queue = asyncio.Queue(maxsize=0)
        for batch in batches:
            match method:
                case "chat":
                    await queue.put(await self.generate_chat_response(
                        batch, model=model, api_key=api_key))
                case "completions":
                    await queue.put(await self.generate_completions_response(
                        batch, model=model, api_key=api_key))

        if not queue.empty():
            return f'Items in Queue: {queue.qsize()} \\n {[await queue.get()]}'
        return "Something went wrong"

    @classmethod
    def generate(cls, prompt: str, *, method: Optional[str]=None, api_key: Optional[str]=None, token_size: Optional[int]=None, model: str, get:str) -> Union[str, ValueError]:
        """
        Generate a response from the OpenAI API based on the specified parameters.

        Args:
            prompt (str): The prompt to be used for generating the response.
            method (Optional[str]): The method to be used for generating the response. Can be either "chat" or "completions".
            api_key (Optional[str]): The API key to be used for generating the response. If not provided, the value of the OPENAI_API_KEY environment variable will be used.
            token_size (Optional[int]): The size of a text batch in tokens.
            model (str): The model to be used for generating the response.
            get (str): The type of response to generate. Can be either "chat", "completions", or "image".

        Returns:
            Union[str, ValueError]: The generated response or a ValueError if the input parameters are invalid.
        """
        if get is None:
            return ValueError("type must be specified either 'chat or completions or image'")

        match get:
            case "chat":
                return asyncio.run(cls().generate_chat_response(prompt, model=model, api_key=api_key))
            case "completions":
                return asyncio.run(cls().generate_completetion_response(prompt, model=model, api_key=api_key))
            case "image":
                return asyncio.run((cls().generate_image(prompt, model=model, api_key=api_key)))
            case "batches":
                return asyncio.run(cls().generate_batches(prompt, method=method, model=model, api_key=api_key ,token_size=token_size))
