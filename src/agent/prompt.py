from transformers import Qwen2VLProcessor, AutoProcessor
from qwen_vl_utils import process_vision_info
import json


# SYSTEM_PROMPT_EN = '''You are a helpful assistant. Answer in English proper. Imagine you are a content moderator on facebook you need to reasoning \
# to category the content of the post (contain an image and a ###Caption) is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm.'''


# # multi-sarcasm sample
# FEW_SHOT_IMAGE_1 = "data/warn_up/warmup-images/75c2dd020173567a242ad1d2f1bd774844832dd2fab51d0663b2d7f58afbc88e.jpg"
# FEW_SHOT_CAPTION_1 = '''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain Sarcasm meaning:
# ### Caption:
# may mà gặp được tôi

# ### OCR (text in the image)
# TRỜI ƠI. LÀM SAO THẾ NÀY \ncậu BẠN NÀY ĐANG VẼ DỞ THÌ LĂN ĐÙNG RA co GIẬT SÙI BỌT MÉP:\nNGUY QUÁ: ĐỀ TÔI GIÚP.
# '''
# FEW_SHOT_REASONING_1 = '''
# ### Reasoning
# Step 1: Describe the image
# This image is a three-panel comic strip featuring three characters. The first character is a person with yellow helmet. \
# The second character is a blue, round bird with a single eye and a small hat. \
# The third character has white hair wearing red pant purpple T-shirt.
# In the first panel, the person exclaims, "Trời ơi! Làm sao thế này?!" and try to helping someone. The blue bird is passing towards the two person.
# In the second panel, there a drawing on the easel, which appears to be a tiger. The yellow helmet person holding white hair person. \
# The person yellow helmet person says, "Cậu bạn này đang vẽ dở thì lăn đùng ra co giật sùi bọt mép".\
# The blue bird responds, "Nguy quá! Để tôi giúp!" (Dangerous! Let me help!).
# In the third panel, the blue bird is standing in front of the easel, looking at the drawing, while the person looks on with a concerned expression.

# Step 2: Evaluate each fact in ### Sarcasm-signal for the image, OCR and the Caption (explain)-> the reason
# The character in the second pannel is falling in the floor and need help but when the blue bird run toward the two person,\
# we expect the blue bird will help the white hair person which appear more emergency but the blue bird just go there and help drawing the unfinish picture. The Sarcasm signal appear is:\
# 2. Contains a mismatch between the text and the image that suggests sarcasm through contradiction or exaggeration. (explain)-> the normaly expect the the blue bird will help white hair person but the blue bird help finish the picture.
# 5. Characters whose actions are absurd and different from what is normally expected. (explain)-> the Caption show out the blue bird will help the white hair person but at the end it not help.

# Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.
# Both image, text in image and Caption contain Sarcasm.
# '''

# # not-sarcasm sample
# FEW_SHOT_IMAGE_2 = "data/warn_up/warmup-images/2e1c147650933097225e804223a966d26c91bc5a92dc600961ed59027d359446.jpg"
# FEW_SHOT_CAPTION_2 = '''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain Sarcasm meaning:

# ### Caption:
# Sơn Tùng M-TP tại Phố đi bộ mang cả \"xe vali gấu bông\" tặng khán giả ♥️♥️♥️

# ### OCR (text in the image)
# Không có văn bản hay chữ trong ảnh
# '''
# FEW_SHOT_REASONING_2 = '''
# ### Reasoning
# Step 1: Describe the image
# The picture shows a person walking on a runway during a fashion show. The individual is wearing a black top with sheer sleeves and red leather pants. \
# They are holding a fur accessory in one hand and have a confident posture. The background is dark, with a spotlight illuminating the model, creating a dramatic effect.

# Step 2: Evaluate each fact in ### not-Sarcasm-signal for the image, OCR and the Caption (explain)-> the reason
# The Caption and image to notify a event organized by a celebrity. The not-Sarcasm signal appear is:
# 1. Conveys sentiments or statements that are straightforward and meant to be taken at face value. (explain)-> The Caption are straightforward to meaning that notify a event organized by a celebrity (Sơn Tùng M-TP) giveaway to his audiences teddy bears
# 2. Aligns directly with the image, supporting the literal interpretation of the text. (explain)-> The Caption mention about Sơn Tùng M-TP and the image also show the picture of him.
# 3. Does NOT contain linguistic or visual cues typically associated with sarcasm. (explain)-> There is not trolling icon or image in the picture, Caption and OCR text

# Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.
# Both image and ###Caption not contain Sarcasm.'''


# REASONING_INS_EN = '''### Sarcasm-signal
# Given Sarcasm is any sample that contain one or more signs in the story (contain text and image) given below:
# 1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
# Made in order to hurt someone's feelings or to criticize something in a humorous way
# 2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
# 3.Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
# 4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
# Text in Caption or image that not conversation but put inside \"\" usually to sarcasm or say opposition thing.
# 5. Characters whose actions are absurd and different from what is normally expected.

# ### not-Sarcasm-signal
# Given not-Sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
# 1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
# 2. Aligns directly with the image, supporting the literal interpretation of the text.
# 3. Does NOT contain linguistic or visual cues typically associated with sarcasm.

# ### Instruction
# Reasoning why this facebook post contain {multi_label} meaning based on the definition about Sarcasm and not-Sarcasm given above. \
# The sample given will be inorder:
# ### Image

# ### Caption

# ### OCR (text in the image)

# ### Reasoning

# Do step-by-step:
# Step 1: Describe the picture.
# Step 2: Evaluate each fact in ### {label}-signal for the image, OCR and the Caption (explain)-> the reason
# Step 2: Reasoning -> conclude for the fact in ### {label}-signal for the image, OCR and the Caption
# Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.'''


# INS_REASONING = '''Reasoning step-by-step the content of this facebook post contain an image and a ###Caption. Do step-by-step:
# 1. Describe the picture.
# 2. Evaluate each fact in Sarcasm sign and not-Sarcasm the in the image and the ###Caption: and explain the reason
# 3. Conclude whether the image and Caption: contain Sarcasm base on evaluation on step 2. and conclude the sample is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm.

# ###Reasoning
# {reasoning}'''

SYSTEM_PROMPT_EN = '''You are a helpful assistant. Answer in English proper. Imagine you are a content moderator on facebook you need to reasoning \
to category the content of the post (contain an image and a ###Caption) is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm.'''


# multi-sarcasm sample
FEW_SHOT_IMAGE_1 = "data/warn_up/warmup-images/75c2dd020173567a242ad1d2f1bd774844832dd2fab51d0663b2d7f58afbc88e.jpg"
FEW_SHOT_CAPTION_1 = '''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain Sarcasm meaning:
### Caption:
may mà gặp được tôi

### OCR (text in the image)
TRỜI ƠI. LÀM SAO THẾ NÀY \ncậu BẠN NÀY ĐANG VẼ DỞ THÌ LĂN ĐÙNG RA co GIẬT SÙI BỌT MÉP:\nNGUY QUÁ: ĐỀ TÔI GIÚP.
'''
FEW_SHOT_REASONING_1 = '''
### Reasoning
Step 1: Describe the image
This image is a three-panel comic strip featuring three characters. The first character is a person with yellow helmet. \
The second character is a blue, round bird with a single eye and a small hat. \
The third character has white hair wearing red pant purpple T-shirt.
In the first panel, the person exclaims, "Trời ơi! Làm sao thế này?!" and try to helping someone. The blue bird is passing towards the two person.
In the second panel, there a drawing on the easel, which appears to be a tiger. The yellow helmet person holding white hair person. \
The person yellow helmet person says, "Cậu bạn này đang vẽ dở thì lăn đùng ra co giật sùi bọt mép".\
The blue bird responds, "Nguy quá! Để tôi giúp!" (Dangerous! Let me help!).
In the third panel, the blue bird is standing in front of the easel, looking at the drawing, while the person looks on with a concerned expression.

Step 2: Give argument then conclude one or more of these (image, OCR or the Caption) contain on or more facts in ### Sarcasm-signal
The character in the second pannel is falling in the floor and need help but when the blue bird run toward the two person,\
we expect the blue bird will help the white hair person which appear more emergency but the blue bird just go there and help drawing the unfinish picture. The Sarcasm signal appear is:\
2. the normaly expect the the blue bird will help white hair person but the blue bird help finish the picture. -> Contains a mismatch between the text and the image that suggests sarcasm through contradiction or exaggeration.
5. the Caption show out the blue bird will help the white hair person but at the end it not help. -> Characters whose actions are absurd and different from what is normally expected.

Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.
Both image, text in image and Caption contain Sarcasm.
'''

# not-sarcasm sample
FEW_SHOT_IMAGE_2 = "data/warn_up/warmup-images/2e1c147650933097225e804223a966d26c91bc5a92dc600961ed59027d359446.jpg"
FEW_SHOT_CAPTION_2 = '''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain Sarcasm meaning:

### Caption:
Sơn Tùng M-TP tại Phố đi bộ mang cả \"xe vali gấu bông\" tặng khán giả ♥️♥️♥️

### OCR (text in the image)
Không có văn bản hay chữ trong ảnh
'''
FEW_SHOT_REASONING_2 = '''
### Reasoning
Step 1: Describe the image
The picture shows a person walking on a runway during a fashion show. The individual is wearing a black top with sheer sleeves and red leather pants. \
They are holding a fur accessory in one hand and have a confident posture. The background is dark, with a spotlight illuminating the model, creating a dramatic effect.

Step 2: Give argument then conclude one or more of these (image, OCR or the Caption) contain on or more facts in ### not-Sarcasm-signal
The Caption and image to notify a event organized by a celebrity. The not-Sarcasm signal appear is:
1. The Caption are straightforward to meaning that notify a event organized by a celebrity (Sơn Tùng M-TP) giveaway to his audiences teddy bears -> Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. The Caption mention about Sơn Tùng M-TP and the image also show the picture of him. -> Aligns directly with the image, supporting the literal interpretation of the text.
3. There is not trolling icon or image in the picture, Caption and OCR text -> Does NOT contain linguistic or visual cues typically associated with sarcasm.

Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.
Both image and ###Caption not contain Sarcasm.'''


REASONING_INS_EN = '''### Sarcasm-signal
Given Sarcasm is any sample that contain one or more signs in the story (contain text and image) given below:
1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
Made in order to hurt someone's feelings , criticize or to express something in a humorous way
2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
3.Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
Text in Caption or image that not conversation but put inside \"\" usually to sarcasm or say opposition thing.
5. Characters whose actions are absurd and different from what is normally expected.

### not-Sarcasm-signal
Given not-Sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. Aligns directly with the image, supporting the literal interpretation of the text.
3. Does NOT contain linguistic or visual cues typically associated with sarcasm.

### Instruction
Reasoning why this facebook post contain {multi_label} meaning based on the definition about Sarcasm and not-Sarcasm given above. \
The sample given will be inorder:
### Image

### Caption

### OCR (text in the image)

### Reasoning

Do step-by-step:
Step 1: Describe the picture.
Step 2: Give argument then conclude one or more of these (image, OCR or the Caption) contain on or more facts in ### {label}-signal
Step 3: Conclude whether the image, OCR and Caption contain Sarcasm or not-Sarcasm base on evaluation on step 2.'''


INS_REASONING = '''Reasoning step-by-step the content of this facebook post contain an image and a ###Caption. Do step-by-step:
1. Describe the picture.
2. Evaluate each fact in Sarcasm sign and not-Sarcasm the in the image and the ###Caption: and explain the reason
3. Conclude whether the image and Caption: contain Sarcasm base on evaluation on step 2. and conclude the sample is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm.

### Reasoning
{reasoning}'''

import base64
import requests
from io import BytesIO
from PIL import Image


def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


def create_reas_prompt_api(
        processor:Qwen2VLProcessor,
        image_path,
        caption,
        label,
        ocr,
):
    MAP = {
        "multi-sarcasm": "Sarcasm",
        "text-sarcasm": "Sarcasm",
        "image-sarcasm": "Sarcasm",
        "not-sarcasm": "not-Sarcasm",
    }
    img_1 = Image.open(FEW_SHOT_IMAGE_1)
    base64_img_1 = encode_image(img_1)
    img_2 = Image.open(FEW_SHOT_IMAGE_2)
    base64_img_2 = encode_image(img_2)
    img_3 = Image.open(image_path)
    base64_img_3 = encode_image(img_3)

    api = "https://api.hyperbolic.xyz/v1/chat/completions"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqdWxpdXMuMjAwMy5uby4yQGdtYWlsLmNvbSIsImlhdCI6MTcyOTA5MzY4NX0.hNgHQBDy2FdBZmpLnvg7XiH9rsKSgPXVTC69Jlw_JWk"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    message = [
        {"role": "system", "content": SYSTEM_PROMPT_EN},
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": REASONING_INS_EN.format(multi_label="multi-sarcasm", label="Sarcasm")},
        #         {
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/jpeg;base64,{base64_img_1}"},
        #         },
        #         {"type": "text", "text": FEW_SHOT_CAPTION_1}
        #     ]
        # },
        # {"role": "assistant", "content": FEW_SHOT_REASONING_1},
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": REASONING_INS_EN.format(multi_label="not-Sarcasm", label="not-Sarcasm")},
        #         {
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/jpeg;base64,{base64_img_2}"},
        #         },
        #         {"type": "text", "text": FEW_SHOT_CAPTION_2},
        #     ]
        # },
        # {"role": "assistant", "content": FEW_SHOT_REASONING_2},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REASONING_INS_EN.format(multi_label=label, label=MAP[label])},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img_3}"},
                },
                {"type": "text", "text": f'''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain {label} meaning:
### Caption:
{caption}

### OCR (text in image)
{ocr}

### Reasoning'''},
            ]
        }
    ]

    payload = {
        "messages": message,
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.001,
    }

    response = requests.post(api, headers=headers, json=payload)
    print(response)

    return response


def create_reas_prompt(
        processor:Qwen2VLProcessor,
        image_path,
        caption,
        label,
        ocr,
):
    MAP = {
        "multi-sarcasm": "Sarcasm",
        "text-sarcasm": "Sarcasm",
        "image-sarcasm": "Sarcasm",
        "not-sarcasm": "not-Sarcasm",
    }
    message = [
        {"role": "system", "content": SYSTEM_PROMPT_EN},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REASONING_INS_EN.format(multi_label="multi-sarcasm", label="Sarcasm")},
                {"type": "image", "image": FEW_SHOT_IMAGE_1},
                {"type": "text", "text": FEW_SHOT_CAPTION_1}
            ]
        },
        {"role": "assistant", "content": FEW_SHOT_REASONING_1},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REASONING_INS_EN.format(multi_label="not-Sarcasm", label="not-Sarcasm")},
                {"type": "image", "image": FEW_SHOT_IMAGE_2},
                {"type": "text", "text": FEW_SHOT_CAPTION_2},
            ]
        },
        {"role": "assistant", "content": FEW_SHOT_REASONING_2},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REASONING_INS_EN.format(multi_label=label, label=MAP[label])},
                {"type": "image", "image": image_path},
                {"type": "text", "text": f'''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain {label} meaning:
### Caption:
{caption}

### OCR (text in image)
{ocr}'''},
            ]
        }
    ]

    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs

if __name__ == "__main__":
    from transformers import Qwen2VLForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    inputs = create_reas_prompt(
        processor,
        "data/warn_up/warmup-images/bc24654fb4fba69b41b6b4dce15295fc4acc8ebce9b9bff452ef6a8890e04e72.jpg",
        "No shjt Sherlock\n#interpool",
        "multi-sarcasm",
        "Tiền Phong - 2 giờ\nPhát hiện hai xác chết trong nghĩa trang",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024#, temperature=0.2, top_k=2, top_p=0.2, repetition_penalty=1
                                   )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    
    # output = create_reas_prompt(
    #     processor,
    #     "data/warn_up/warmup-images/bc24654fb4fba69b41b6b4dce15295fc4acc8ebce9b9bff452ef6a8890e04e72.jpg",
    #     "No shjt Sherlock\n#interpool",
    #     "multi-sarcasm",
    #     "Tiền Phong - 2 giờ\nPhát hiện hai xác chết trong nghĩa trang",
    # )

    # print("############################")
    # print(output.json)