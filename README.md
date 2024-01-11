# Interface to OpenAI API
The OpenAIAPI class provides an interface to the OpenAI API. It provides methods for cleaning text, generating responses, and generating images. The class can be used to generate responses in parallel, and it supports multiple models and methods for generating responses. The class also provides a generate method that can be used to generate responses based on the specified parameters.

## Table of content
1) [Purpose](#purpose)
2) [About](#about)
3) [Authorship](#authorship)
4) [Methods](#methods)
5) [Examples of use](#examples-of-use)
6) [Fixes](#fixes)


## Purpose 
This project was developed for personal use, functioning as a sort of all-in-one class. It encapsulates a variety of functions and features that I found useful and needed frequently. 

## About
This versatile class presents several options for interaction, including chat, completions, and image end points. To ensure a friendly and respectful environment, all input text undergoes a profanity check and any profane words are censored. 
In order to optimize efficiency, any input text that exceeds 30 words will undergo stop word removal to decrease its size. The class is designed to locate a `.env` file containing the `'OPENAI_API_KEY'` variable. This file should be located in the directory of the current script. 
Alternatively, you may pass the API key when initializing the class. This class offers a method that allows to split a very long prompt like an article or a book into multiple responses in parallel using the OpenAI API.

## Authorship
This project was created by Dennis Rotnov. Feel free to use, reuse and modify it, but please retain this authorship attribution.

## Methods
text_cleaner:
Cleans the text by removing stopwords and censoring profane words.
```python
def text_cleaner(self, prompt: str) -> str:
        """
        Cleans the text by removing stopwords to make prompt shorter and censoring profane words.

        Args:
            prompt (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        english_stopwords = stopwords.words('english')
        tokenized_words = prompt.split(" ")
        if len(tokenized_words) > 30:
            prompt = " ".join([word for word in tokenized_words if word not in english_stopwords])
            return profanity.censor(prompt)
        return profanity.censor(prompt)
```
batched_prompt:
Splits a long prompt into batches of certain length expressed in tokens.
```python
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
        total_tokens = math.ceil(len(prompt.split(" "))*1.4)
        batches = math.ceil(total_tokens / token_size)
        if total_tokens > token_size:
            for batch in batched(prompt.split(" "), math.ceil(len(prompt.split(" "))/batches)):
                yield " ".join(batch)
        yield prompt
```
generate_chat_response:
Generates a response from the OpenAI API based on the chat model.
```python
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
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            print(f"Error while fetching response. Error: {e}")
        else: return completion.choices[0].message.content

```
generate_completetion_response:
Generates a completion response from the OpenAI API based on the completion model.
```python
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
        try:
            completions = await client.completions.create(
                model=model,
                prompt=prompt,
                n=1,
                top_p=1,
                stop=None,
                temperature=0.5,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
        except Exception as e:
            print(f"Error while fetching response. Error: {e}")
        else: return completions.choices[0].text

```
generate_image:
Generates an image from the OpenAI API based on the image model.
```python
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
        try:
            response = await client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
        except Exception as e:
            print(f"Error while fetching response. Error: {e}")
        else: return image_url
```
generate_batches:
Generates multiple responses in parallel using the OpenAI API.
```python
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
        # await queue.join()
        if not queue.empty():
            return f'Items in Queue: {queue.qsize()} \\n {[await queue.get() for _ in range(queue.qsize())]}'
        return "Something went wrong"
```
generate:
Generates a response from the OpenAI API based on the specified parameters.
```python
 @classmethod
    def generate(cls, prompt: str, *, method: Optional[str]=None, api_key: Optional[str]=None, token_size: Optional[int]=None, model: Optional[str]=None, get: Optional[str]=None) -> str:
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
            raise ValueError("Type must be specified either 'chat or completions or image'")
        if model is None: raise ValueError("Model must be specified")

        match get:
            case "chat":
                return asyncio.run(cls().generate_chat_response(prompt, model=model, api_key=api_key))
            case "completions":
                return asyncio.run(cls().generate_completetion_response(prompt, model=model, api_key=api_key))
            case "image":
                return asyncio.run((cls().generate_image(prompt, model=model, api_key=api_key)))
            case "batches":
                return asyncio.run(cls().generate_batches(prompt, method=method, model=model, api_key=api_key ,token_size=token_size))

```

## Examples of use
Very long input that needs to be split in multiple inputs. You specify how you want to split prompt by providing the length of desired input length expressed in tokens.
```python
prompt = """SUMMARIZE: ...A VERY LONG TEXT..."""
res = OpenAIAPI.generate(
   prompt, get='batches', method="chat", model="gpt-3.5-turbo-1106", token_size=4000)
print(res)
```

Chat end point.
```python
res = OpenAIAPI.generate(
    'Say hello', get='chat', model="gpt-3.5-turbo-1106")
print(res)
```

Completions endpoint.
```python
res = OpenAIAPI.generate(
    'Say hello to me', get='completions', model="gpt-3.5-turbo-instruct")
print(res)
```

Generate image end point.
```python
res = OpenAIAPI.generate(
    'Cat', get='image', model="dall-e-3")
print(res)
```

## Fixes
####  1/11/24 Fixed a bug related to counting tokens tokes. Now uses regular expressions and new multiplier for convirsion.
#### Updated generate_batches method. Now runs batches. joins result and provides a single summary.
 
