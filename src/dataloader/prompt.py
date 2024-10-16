SYSTEM_PROMPT = "You are a helpful assistant. Imagine you are a content moderator on facebook you need to reasoning\
to category the content of the post (contain an image and a caption) is multi-sarcasm, non-sarcasm, image-sarcasm or text-sarcasm"

USER_PROMPT = """You need to classify which post is multi-sarcasm, non-sarcasm, image-sarcasm, text-sarcasm.\
Sarcasm is any sample that satisfy any condition below:
1. Employs irony by saying the opposite of what is meant, especially to
mock or deride.
2. Contains a mismatch between the text and the image that suggests
sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is
clearly not meant to be taken literally.
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are
commonly used to convey sarcasm online.

The post you need to classify contain the image above. The caption of the post is {caption}. \
Text the image is: {ocr}.""" 