from openai import OpenAI

class Llm:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def construct_message(user_msg, sys_msg=None):
        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_msg})
        return messages

    def ask(self, user_msg, sys_msg=None, schema=None,
            temperature=None, top_p=None):
        messages = self.construct_message(user_msg, sys_msg)

        llm_args = {'model': 'gpt-5.1', 'messages': messages}
        if temperature:
            llm_args['temperature'] = temperature
        if top_p:
            llm_args['top_p'] = top_p

        if schema:
            completion = self.client.chat.completions.parse(
                response_format=schema,
                **llm_args
            )
            return completion.choices[0].message.parsed
        else:
            completion = self.client.chat.completions.create(**llm_args)
            return completion.choices[0].message.content
