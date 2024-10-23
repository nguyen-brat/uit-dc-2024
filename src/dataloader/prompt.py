SYSTEM_PROMPT = "You are a helpful assistant. Imagine you are a content moderator on facebook you need to reasoning\
to category the content of the post (contain an image and a caption) is multi-sarcasm, non-sarcasm, image-sarcasm or text-sarcasm"

USER_PROMPT = """Imagine you are a content moderator on facebook you need to reasoning \
to category the content of the post (contain an image and a caption) is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm. \
Suggest some sarcasm and not-sarcasm singals:

### sarcasm-signal
Given sarcasm is any sample that contains one or more signs in the story (containing text and image) given below:
1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
Made in order to hurt someone's feelings, criticize or express something in a humorous way
2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
Text in Caption or image that is not a conversation but put inside \"\" usually to sarcasm or say opposition thing.
5. Characters whose actions are absurd and different from what is normally expected.

### not-sarcasm-signal
Given not-sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. Aligns directly with the image, supporting the literal interpretation of the text.
3. Does NOT contain linguistic or visual cues typically associated with sarcasm.

Given a Facebook post contains an image <image>, the reference text in the image is: {ocr} and a caption: {caption}. \
Explain step-by-step by analysis the image and caption then give the conclusion that is post have multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm meaning."""