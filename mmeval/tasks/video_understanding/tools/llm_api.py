import openai
openai.api_key = ''

def ask(input_text, system=None):
    # Compute the correctness score
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    # Convert response to a Python dictionary.
    response_message = completion["choices"][0]["message"]["content"]
    # response_dict = ast.literal_eval(response_message)
    return response_message
