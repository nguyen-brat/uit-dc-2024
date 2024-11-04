from transformers import Qwen2VLProcessor, AutoProcessor, PreTrainedTokenizer
from qwen_vl_utils import process_vision_info
import json
from PIL import Image
from typing import List
from utils import *


SYSTEM_PROMPT_EN = '''You are a helpful assistant. Answer in English proper. Imagine you are a Vietnamese content moderator on Vietnam facebook post you need to reasoning \
to category the content of the post (contain an image and a caption) is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm.'''


# multi-sarcasm sample
FEW_SHOT_IMAGE_1 = "data/warn_up/warmup-images/75c2dd020173567a242ad1d2f1bd774844832dd2fab51d0663b2d7f58afbc88e.jpg"
FEW_SHOT_CAPTION_1 = '''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain Sarcasm meaning:
### Caption:
may mà gặp được tôi

### OCR (text in the image)
TRỜI ƠI. LÀM SAO THẾ NÀY \ncậu BẠN NÀY ĐANG VẼ DỞ THÌ LĂN ĐÙNG RA co GIẬT SÙI BỌT MÉP:\nNGUY QUÁ: ĐỀ TÔI GIÚP.'''
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
Không có văn bản hay chữ trong ảnh'''

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

SIGNAL = '''Suggest some sarcasm signal and not-sarcasm signal
### Sarcasm-signal
Given Sarcasm is any sample that contains one or more signs in the story (containing text and image) given below:
1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
Made in order to hurt someone's feelings, criticize or express something in a humorous way
2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
Text in Caption or image that is not a conversation but put inside \"\" usually to sarcasm or say opposition thing.
5. Characters whose actions are absurd and different from what is normally expected.

### not-Sarcasm-signal
Given not-Sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. Aligns directly with the image, supporting the literal interpretation of the text.
3. Does NOT contain linguistic or visual cues typically associated with sarcasm.
'''

REASONING_INS_EN = '''### Sarcasm-signal
Given Sarcasm is any sample that contains one or more signs in the story (containing text and image) given below:
1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
Made in order to hurt someone's feelings, criticize or express something in a humorous way
2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
Text in Caption or image that is not a conversation but put inside \"\" usually to sarcasm or say opposition thing.
5. Characters whose actions are absurd and different from what is normally expected.

### not-Sarcasm-signal
Given not-Sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. Aligns directly with the image, supporting the literal interpretation of the text.
3. Does NOT contain linguistic or visual cues typically associated with sarcasm.

### Instruction
Reasoning why this Vietnamese Facebook post contains {multi_label} meaning based on the definition of Sarcasm and not-Sarcasm given above. \
The sample given will be in order:
### Image

### Caption

### OCR (text in the image)

### Reasoning

Do step-by-step:
Step 1: Describe the picture.
Step 2: Give an argument then conclude image, OCR or the Caption contain {label} meaning based on ### {label}-signal given above
Step 3: Conclude whether the image, OCR, and Caption contain Sarcasm or not-Sarcasm based on evaluation in step 2.'''


TRAIN_USER_INS = '''Imagine you are a content moderator on facebook you need to reasoning \
to category the content of the post (contain an image and a caption) is multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm. \
Suggest some sarcasm signal and not-sarcasm :

### Sarcasm-signal
Given Sarcasm is any sample that contains one or more signs in the story (containing text and image) given below:
1. Employs irony by saying the opposite of what is meant, especially to mock or deride. \
Made in order to hurt someone's feelings, criticize or express something in a humorous way
2. Contains a mismatch between the text (in Caption or OCR) and the image that suggests sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is clearly not meant to be taken literally
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are commonly used to convey sarcasm online. \
Text in Caption or image that is not a conversation but put inside \"\" usually to sarcasm or say opposition thing.
5. Characters whose actions are absurd and different from what is normally expected.

### not-Sarcasm-signal
Given not-Sarcasm is any samples that contain one or more signs in the story (contain text and image) given below:
1. Conveys sentiments or statements that are straightforward and meant to be taken at face value.
2. Aligns directly with the image, supporting the literal interpretation of the text.
3. Does NOT contain linguistic or visual cues typically associated with sarcasm.

Given a Facebook post contains an image <image>, the reference text in the image is: {ocr} and a caption: {caption}. \
Explain step-by-step by analysis the image and caption then give the conclusion that is post have multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm meaning. The post is classified as:
- text-sarcasm: if only the content in the caption is sarcastic and the photo and text in the photo are not sarcastic or supportive and have the opposite meaning of the caption.
- image-sarcasm: if the photo or content in the photo contains signs of sarcasm, criticism, belittling or making fun of someone and the caption provided does not contain any signs of sarcasm and has all the elements of not-sarcasm.
- multi-sarcasm: if both the caption and the photo of the post contain elements of sarcasm or the caption supports the semantics of the photo's sarcasm, which can be text in the photo and vice versa.
- not-sarcasm: if both the caption and the photo do not contain signs of sarcasm as provided above. The post has a clear meaning without sarcasm and has the signs of not-sarcasm provided above.'''

TRAIN_SYS_RESPONSE = '''{reason}

**Final-result**
The post is -> {label}'''

VI_REASONING_INSTRUCTION = '<image>\nCho bài đăng trên facebook gồm bức ảnh được cung cấp với dòng caption: {caption}. Bài đăng trên có mang tính châm biếm không và giải thích lí do?'
VI_REASONING_INSTRUCTION_IMAGE = '<image>\nCho bài đăng trên facebook. Nội dung bức ảnh bao gồm chữ trong ảnh có mang tính châm biếm không và vì sao?'

from PIL import Image

def create_reas_prompt_qwen2_vl(
        processor:Qwen2VLProcessor,
        image_paths,
        captions,
        labels,
        ocrs,
):
    MAP = {
        "multi-sarcasm": "multi-sarcasm",
        "text-sarcasm": "text-sarcasm",
        "image-sarcasm": "image-sarcasm",
        "not-sarcasm": "not-Sarcasm",
    }
#     message = [
#         {"role": "system", "content": SYSTEM_PROMPT_EN},
#         # {
#         #     "role": "user",
#         #     "content": [
#         #         {"type": "text", "text": REASONING_INS_EN.format(multi_label="multi-sarcasm", label="Sarcasm")},
#         #         {"type": "image", "image": FEW_SHOT_IMAGE_1},
#         #         {"type": "text", "text": FEW_SHOT_CAPTION_1}
#         #     ]
#         # },
#         # {"role": "assistant", "content": FEW_SHOT_REASONING_1},
#         # {
#         #     "role": "user",
#         #     "content": [
#         #         {"type": "text", "text": REASONING_INS_EN.format(multi_label="not-Sarcasm", label="not-Sarcasm")},
#         #         {"type": "image", "image": FEW_SHOT_IMAGE_2},
#         #         {"type": "text", "text": FEW_SHOT_CAPTION_2},
#         #     ]
#         # },
#         # {"role": "assistant", "content": FEW_SHOT_REASONING_2},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": REASONING_INS_EN.format(multi_label=label, label=MAP[label])},
#                 {"type": "image", "image": image_path},
#                 {"type": "text", "text": f'''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain {label} meaning step-by-step:
# ### Caption:
# {caption}

# ### OCR (text in image)
# {ocr}'''},
#             ]
#         }
#     ]

    messages = [
            [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":SIGNAL},
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f'''Given a Facebook post contains an image below and a caption: {caption}. The text in the image is: {ocr}. Explain step-by-step why this post contain {label} meaning ?'''},
                ]
            }
        ] for image_path, caption, ocr, label in zip(image_paths, captions, ocrs, labels)
    ]


    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        ) for message in messages
    ]
    

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def create_reas_prompt_pixtral(
        processor,
        image_paths:List[str],
        captions,
        labels,
        ocrs,
):
    # message = [
    #     {"role": "system", "content": SYSTEM_PROMPT_EN},
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "content": REASONING_INS_EN.format(multi_label="multi-sarcasm", label="Sarcasm")},
        #         {"type": "image"},
        #         {"type": "text", "content": FEW_SHOT_CAPTION_1}
        #     ]
        # },
        # {"role": "assistant", "content": FEW_SHOT_REASONING_1},
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "content": REASONING_INS_EN.format(multi_label="not-Sarcasm", label="not-Sarcasm")},
        #         {"type": "image"},
        #         {"type": "text", "content": FEW_SHOT_CAPTION_2},
        #     ]
        # },
        # {"role": "assistant", "content": FEW_SHOT_REASONING_2},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "content": REASONING_INS_EN.format(multi_label=label, label=MAP[label])},
#                 {"type": "image"},
#                 {"type": "text", "content": f'''Given the content of a facebook post with image above, Caption and OCR (text in image) below contain {label} meaning:
# ### Caption:
# {caption}

# ### OCR (text in image)
# {ocr}'''},
#             ]
#         }
#     ]


    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": [
                    {"type": "text", "content":SIGNAL},
                    {"type": "image"},
                    {"type": "text", "content": f"Given a Facebook post contains an image above and a caption: {caption}. The text in the image is: {ocr}. Explain step-by-step why this post contain {label} meaning ?"},
                ]
            }
        ] for caption, ocr, label in zip(captions, ocrs, labels)
    ]

    # inputs = processor(text=prompt, images=[Image.open(FEW_SHOT_IMAGE_1), Image.open(FEW_SHOT_IMAGE_2), Image.open(image_path)], return_tensors="pt")
    prompts = [processor.apply_chat_template(message) for message in messages]
    inputs = processor(
        text=prompts,
        images=[[Image.open(image_path)] for image_path in image_paths],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    return inputs


def create_vi_intern_prompt(
        tokenizer:PreTrainedTokenizer,
        image_paths:List[str],
        captions:List[str],
        labels:List[str],
        ocrs:List[str],
):
    pixel_values = load_image(image_paths[0], max_num=6).to(torch.bfloat16).cuda()
    question = VI_REASONING_INSTRUCTION.format(caption=captions[0])
    return dict(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question
    )


def create_vi_intern_prompt_image(
        tokenizer:PreTrainedTokenizer,
        image_paths:List[str],
        captions:List[str],
        labels:List[str],
        ocrs:List[str],
):
    pixel_values = load_image(image_paths[0], max_num=6).to(torch.bfloat16).cuda()
    question = VI_REASONING_INSTRUCTION_IMAGE
    return dict(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question
    )


if __name__ == "__main__":
    # from transformers import Qwen2VLForConditionalGeneration

    # processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )

    # inputs = create_reas_prompt_qwen2_vl(
    #     processor,
    #     "data/warn_up/warmup-images/bc24654fb4fba69b41b6b4dce15295fc4acc8ebce9b9bff452ef6a8890e04e72.jpg",
    #     "No shjt Sherlock",
    #     "multi-sarcasm",
    #     "Tiền Phong - 2 giờ\nPhát hiện hai xác chết trong nghĩa trang",
    # ).to(model.device)

    # inputs = create_reas_prompt_qwen2_vl(
    #     processor,
    #     "data/warn_up/warmup-images/a24d92ad80a7598313c1ce08769160cb2a3f9a18b06af0fe40e3aeca71c508e3.jpg",
    #     "Không biết nói tiếng Anh thì bảo người ta nói tiếng Việt, thế mà trước giờ không nghĩ ra",
    #     "multi-sarcasm",
    #     "MÊ CÁCH NGỌC TRINH GIAO TIẾP VỚI TRAI TÂY: \"EM NGHĨ LÀ TỐT NHẤT ANH NÊN NÓI TIẾNG VIỆT ĐI HA, NÓI MỘT HỒI LÀ EM KHÔNG HIỂU GÌ HẾT\"",
    # ).to(model.device)

    # inputs = create_reas_prompt_qwen2_vl(
    #     processor,
    #     "data/warn_up/warmup-images/34bbe4d8a417117023762ca6a80c11879e7823cd41f72a78863fe8a2623cda95.jpg",
    #     "Việt Nam, map game sinh tồn khó nhất thế giới.\n-The Deep",
    #     "multi-sarcasm",
    #     "Bức ảnh là một trang web của báo Người Lao Động với tiêu đề \"NGƯỜI LAO ĐỘNG\" được in đậm và cỡ chữ lớn ở đầu trang. Bên dưới tiêu đề là thanh menu với các mục: \"TRONG NƯỚC\", \"QUỐC TẾ\", \"CÔNG ĐOÀN\", \"BAN ĐỌC\", \"KINH TẾ\", \"SỨC KHỎE\", \"GIÁO DỤC\", \"PHÁP LUẬT\". Tiếp theo là phần nội dung bài viết với tiêu đề \"Chê nhạc \"dở ec\", bị đâm vào chỗ hiểm đến chết!\" được in đậm và cỡ chữ lớn. Dưới tiêu đề là thông tin về thời gian đăng tải bài viết: \"19-01-2013 - 11:07 | Pháp luật\". Phần nội dung bài viết được chia thành nhiều đoạn văn, mỗi đoạn được đánh dấu bằng dấu gạch ngang. Nội dung bài viết xoay quanh vụ việc một người đàn ông tên Võ Hùng Hậu (SN 1986, ngụ quận 3) đã bị Công an quận 3-TPHCM bắt khẩn cấp do có hành vi giết ông Nguyễn Đông Phương (SN 1957, ngụ phường 4, quận 3). Trước đó, ông Phương sang nhà Hậu chơi. Hậu đang treo bạc để dọn hàng bán và có mở nhạc xập xình. Ông Phương nhận xét: \"Nhạc dở ec mà cũng nghe, đã vậy còn mở lớn nữa\". Nghe vậy, Hậu lớn tiếng chửi lại ông Phương rồi cả hai xảy ra ẩu đả. Trong lúc nóng giận, Hậu chụp cây kéo gần đó đâm nhiều nhát khiến ông Phương gục tại chỗ. Thấy ông Phương bị đâm, một số người nhào vô can thì cũng bị Hậu gây thương tích.",
    # ).to(model.device)

    # inputs = create_reas_prompt_qwen2_vl(
    #     processor,
    #     [
    #         "data/public_train/train-images/2b144cb2df4021574d73d82b20a2a5d86e8a11433a6c0fe7617ce0b67a868be1.jpg",
    #         #"data/public_train/train-images/f9a506b8dc0b5e770c31e327a85915a57112772c4966e7a0e77b0974e86b8dcf.jpg",
    #     ],
    #     [
    #         "Chúc gì kì vậy 🤣",
    #         #"Cuộc sống làm mẹ là vợ của người phụ nữ gói trọn trong 1 bức ảnh",
    #     ],
    #     [
    #         "image-sarcasm", 
    #         #"not-sarcasm", 
    #         # "image-sarcasm", "not-sarcasm", "image-sarcasm", "not-sarcasm",
    #     ],
    #     [
    #         "Trích xuất văn bản có trong ảnh:\n- Trả lời bình luận của phantran1609\n- tội chị ghê, chúc chị sớm mọc lại chân nha\n- @Tú Linh\n- Trả lời\n- @phantran1609\n- cắt chân rồi sao mọc ra được nữa e kaka, chúc mn ngày m... Xem thêm\n- Linh Âm thanh Gốc\n- @Tú L",
    #         #"Không có văn bản nào được trích xuất từ hình ảnh này.",
    #     ],
    # ).to(model.device)


    # generated_ids = model.generate(**inputs, max_new_tokens=1024#, temperature=0.2, top_k=2, top_p=0.2, repetition_penalty=1
    #                                )
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text[0])
    # print("================")
    # print(output_text[1])


    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image
    import torch

    model_id = "mistral-community/pixtral-12b"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).eval()

    # inputs = create_reas_prompt_pixtral(
    #     processor,
    #     "data/warn_up/warmup-images/bc24654fb4fba69b41b6b4dce15295fc4acc8ebce9b9bff452ef6a8890e04e72.jpg",
    #     "No shjt Sherlock",
    #     "multi-sarcasm",
    #     "Tiền Phong - 2 giờ\nPhát hiện hai xác chết trong nghĩa trang",
    # ).to(model.device)

    # inputs = create_reas_prompt_pixtral(
    #     processor,
    #     "data/warn_up/warmup-images/a24d92ad80a7598313c1ce08769160cb2a3f9a18b06af0fe40e3aeca71c508e3.jpg",
    #     "Không biết nói tiếng Anh thì bảo người ta nói tiếng Việt, thế mà trước giờ không nghĩ ra",
    #     "multi-sarcasm",
    #     "MÊ CÁCH NGỌC TRINH GIAO TIẾP VỚI TRAI TÂY: \"EM NGHĨ LÀ TỐT NHẤT ANH NÊN NÓI TIẾNG VIỆT ĐI HA, NÓI MỘT HỒI LÀ EM KHÔNG HIỂU GÌ HẾT\"",
    # ).to(model.device)

    # inputs = create_reas_prompt_pixtral(
    #     processor,
    #     "data/warn_up/warmup-images/34bbe4d8a417117023762ca6a80c11879e7823cd41f72a78863fe8a2623cda95.jpg",
    #     "Việt Nam, map game sinh tồn khó nhất thế giới.\n-The Deep",
    #     "multi-sarcasm",
    #     "Bức ảnh là một trang web của báo Người Lao Động với tiêu đề \"NGƯỜI LAO ĐỘNG\" được in đậm và cỡ chữ lớn ở đầu trang. Bên dưới tiêu đề là thanh menu với các mục: \"TRONG NƯỚC\", \"QUỐC TẾ\", \"CÔNG ĐOÀN\", \"BAN ĐỌC\", \"KINH TẾ\", \"SỨC KHỎE\", \"GIÁO DỤC\", \"PHÁP LUẬT\". Tiếp theo là phần nội dung bài viết với tiêu đề \"Chê nhạc \"dở ec\", bị đâm vào chỗ hiểm đến chết!\" được in đậm và cỡ chữ lớn. Dưới tiêu đề là thông tin về thời gian đăng tải bài viết: \"19-01-2013 - 11:07 | Pháp luật\". Phần nội dung bài viết được chia thành nhiều đoạn văn, mỗi đoạn được đánh dấu bằng dấu gạch ngang. Nội dung bài viết xoay quanh vụ việc một người đàn ông tên Võ Hùng Hậu (SN 1986, ngụ quận 3) đã bị Công an quận 3-TPHCM bắt khẩn cấp do có hành vi giết ông Nguyễn Đông Phương (SN 1957, ngụ phường 4, quận 3). Trước đó, ông Phương sang nhà Hậu chơi. Hậu đang treo bạc để dọn hàng bán và có mở nhạc xập xình. Ông Phương nhận xét: \"Nhạc dở ec mà cũng nghe, đã vậy còn mở lớn nữa\". Nghe vậy, Hậu lớn tiếng chửi lại ông Phương rồi cả hai xảy ra ẩu đả. Trong lúc nóng giận, Hậu chụp cây kéo gần đó đâm nhiều nhát khiến ông Phương gục tại chỗ. Thấy ông Phương bị đâm, một số người nhào vô can thì cũng bị Hậu gây thương tích.",
    # ).to(model.device)

    inputs = create_reas_prompt_pixtral(
        processor,
        [
            "data/public_train/train-images/2b144cb2df4021574d73d82b20a2a5d86e8a11433a6c0fe7617ce0b67a868be1.jpg",
            "data/public_train/train-images/f9a506b8dc0b5e770c31e327a85915a57112772c4966e7a0e77b0974e86b8dcf.jpg"
        ],
        [
            "Chúc gì kì vậy 🤣",
            "Cuộc sống làm mẹ là vợ của người phụ nữ gói trọn trong 1 bức ảnh",
        ],
        [
            "image-sarcasm",
            "not-sarcasm"
        ],
        [
            "Trích xuất văn bản có trong ảnh:\n- Trả lời bình luận của phantran1609\n- tội chị ghê, chúc chị sớm mọc lại chân nha\n- @Tú Linh\n- Trả lời\n- @phantran1609\n- cắt chân rồi sao mọc ra được nữa e kaka, chúc mn ngày m... Xem thêm\n- Linh Âm thanh Gốc\n- @Tú L",
            "Không có văn bản nào được trích xuất từ hình ảnh này."
        ],
    ).to(model.device)


    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=500, repetition_penalty= 1.0, temperature= 0.1, top_p=0.005, top_k= 1,)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    print("==========================")
    print(output_text[1])