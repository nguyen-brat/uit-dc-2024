SYSTEM_PROMPT = "You are a helpful assistant. Imagine you are a content moderator on facebook you need to reasoning\
to category the content of the post (contain an image and a caption) is multi-sarcasm, non-sarcasm, image-sarcasm or text-sarcasm."

SYSTEM_PROMPT_V2 = "You are a helpful assistant."

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

USER_PROMPT_V2 = """Imagine you are a content moderator on facebook you need to reasoning \
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

Given a Facebook post contains an image <image>, the describe about the image and reference text in the image is: {ocr} and a caption: {caption}. \
Explain step-by-step by analysis the image and caption then give the conclusion that is post have multi-sarcasm, not-sarcasm, image-sarcasm or text-sarcasm meaning."""


USER_PROMPT_V3 = """Hãy tưởng tượng bạn là người kiểm duyệt nội dung trên facebook, bạn cần lý luận \
để phân loại nội dung bài đăng (có hình ảnh và chú thích) là đa châm biếm, không châm biếm, châm biếm hình ảnh hoặc châm biếm văn bản. \
Gợi ý một số ký hiệu châm biếm và không châm biếm:

### tín hiệu châm biếm
Mỉa mai được đưa ra là bất kỳ mẫu nào có chứa một hoặc nhiều ký hiệu trong câu chuyện (có chứa văn bản và hình ảnh) được đưa ra dưới đây:
1. Sử dụng sự mỉa mai bằng cách nói ngược lại với ý nghĩa, đặc biệt là để chế giễu hoặc giễu cợt. \
Được thực hiện để làm tổn thương cảm xúc của ai đó, chỉ trích hoặc diễn đạt điều gì đó theo cách hài hước
2. Có sự không khớp giữa văn bản (trong Chú thích hoặc OCR) và hình ảnh gợi ý sự châm biếm thông qua sự mâu thuẫn hoặc cường điệu.

3. Sử dụng phép cường điệu để cường điệu hóa hoặc hạ thấp thực tế theo cách rõ ràng là không có nghĩa đen
4. Kết hợp các hashtag, biểu tượng cảm xúc hoặc dấu câu mỉa mai, thường được sử dụng để truyền tải sự mỉa mai trực tuyến. \
Văn bản trong chú thích hoặc hình ảnh không phải là cuộc trò chuyện nhưng được đặt bên trong \"\" thường để mỉa mai hoặc nói điều đối lập.
5. Các nhân vật có hành động vô lý và khác với những gì thường được mong đợi.

### tín hiệu không châm biếm
Không-sarcasm được đưa ra là bất kỳ mẫu nào chứa một hoặc nhiều dấu hiệu trong câu chuyện (chứa văn bản và hình ảnh) được đưa ra dưới đây:
1. Truyền tải tình cảm hoặc tuyên bố thẳng thắn và có nghĩa đen.
2. Phù hợp trực tiếp với hình ảnh, hỗ trợ cho cách diễn giải theo nghĩa đen của văn bản.
3. KHÔNG chứa các tín hiệu ngôn ngữ hoặc hình ảnh thường liên quan đến sự mỉa mai.

Cho một bài đăng trên Facebook có chứa hình ảnh <image>, mô tả về hình ảnh và văn bản tham khảo trong hình ảnh là: {ocr} và chú thích: {caption}. \
Giải thích từng bước bằng cách phân tích hình ảnh và chú thích sau đó đưa ra kết luận rằng bài đăng có ý nghĩa image-sarcasm nếu ảnh hoặc chữ trong ảnh mang ý nghĩa châm biếm \
hoặc text-sarcasm nếu chỉ chú trong bài đăng mang tính châm biếm và multi-sarcasm nếu cả ảnh và chú thích đều mang tính châm biếm, not-sarcasm nếu cả chú thích và ảnh đều không mang tính châm biếm."""

ASSISTANT_ANSWER = """Nhận định về tính châm biếm trong chú thích của bài đăng: {text_reasoning}
Nhận định về tính châm biếm trong hình ảnh của bài đăng: {image_reasoning}
Nhận định về tính chấm biếm tổng quát của bài đăng: {reasoning}"""