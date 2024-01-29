
class BaseModel(object):
    def __init__(self):
        pass

    def history_to_prompt(self, query, history=[]):
        """Concatenate history to query prompts

        Args:
            query (str): question
            history (list, optional): [(q1, a1), (q2, a2), ...]
        """
        pass

    @NotImplementedError
    def generate(self, prompt, image_path, history=[]):
        """

        Args:
            prompt (_type_): _description_
            image_path (_type_): _description_
            history (list, optional): _description_. Defaults to [].
        """
        pass

if __name__ == "__main__":
    basemodel = BaseModel()
    prompt = basemodel.history_to_prompt(
        query="Name of them?",
        history=[("How many people in this image?", "There are two people.")]
    )
    print(prompt)