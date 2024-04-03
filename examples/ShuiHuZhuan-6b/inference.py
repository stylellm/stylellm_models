from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stylellm/ShuiHuZhuan-6b")
model = AutoModelForCausalLM.from_pretrained("stylellm/ShuiHuZhuan-6b").eval()

messages = [{"role": "user", "content": "严冬时节，鹅毛一样的大雪片在天空中到处飞舞着，有一个王后坐在王宫里的一扇窗子边，正在为她的女儿做针线活儿，寒风卷着雪片飘进了窗子，乌木窗台上飘落了不少雪花。"}]
input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print("Output:", response)
# Output: 此时正是隆冬之际，鹅毛般大雪乱飘，有一王后坐于王宫内一扇窗户前，正缝其女衣，寒风卷雪入得窗来，把那乌木窗台都压坏了。
