<div align="center">
<h1>StyleLLM 文风大模型</h1>
</div>

# 模型介绍
 **stylellm** 是一个基于大语言模型（llm）的文本风格迁移（text style transfer）项目。项目利用大语言模型来学习指定文学作品的写作风格（惯用词汇、句式结构、修辞手法、人物对话等），形成了一系列特定风格的模型。  
 
利用**stylellm**模型可将学习到的风格移植至其他通用文本上，即：输入一段原始文本，模型可对其改写，输出带有该风格特色的文本，达到文字修饰、润色或风格模仿的效果。


# 项目内容

- 📚 [四大名著风格模型](#模型列表) ([量化版本](#%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8awq4bits))
- ↔️ [风格转化效果](#风格转化效果)
- 💬 [风格化聊天效果](#风格化聊天效果)


# 模型列表
**stylellm-中国四大名著系列** 包括基于Yi-6b微调的四个模型，分别采用《三国演义》、《西游记》、《水浒传》、《红楼梦》四本中国古典长篇小说训练而成：


| 风格 | 模型 | 使用方法 |
|:----------|:----------|:----------|
| 三国演义风格 | 🤗 [stylellm/SanGuoYanYi-6b](https://huggingface.co/stylellm/SanGuoYanYi-6b)  | [inference.py](https://github.com/stylellm/stylellm_models/blob/main/examples/SanGuoYanYi-6b/inference.py)  |
| 西游记风格 | 🤗 [stylellm/XiYouJi-6b](https://huggingface.co/stylellm/XiYouJi-6b)  | [inference.py](https://github.com/stylellm/stylellm_models/blob/main/examples/XiYouJi-6b/inference.py)  |
| 水浒传风格 | 🤗 [stylellm/ShuiHuZhuan-6b](https://huggingface.co/stylellm/ShuiHuZhuan-6b)  | [inference.py](https://github.com/stylellm/stylellm_models/blob/main/examples/ShuiHuZhuan-6b/inference.py)  |
| 红楼梦风格 | 🤗 [stylellm/HongLouMeng-6b](https://huggingface.co/stylellm/HongLouMeng-6b)  | [inference.py](https://github.com/stylellm/stylellm_models/blob/main/examples/HongLouMeng-6b/inference.py)  |
<br/>

## 量化模型列表（AWQ，4bits）
| 风格 | 模型 | 
|:----------|:----------|
| 三国演义风格 | 🤗 [stylellm/SanGuoYanYi-6b-AWQ](https://huggingface.co/stylellm/SanGuoYanYi-6b-AWQ)  |
| 西游记风格 | 🤗 [stylellm/XiYouJi-6b-AWQ](https://huggingface.co/stylellm/XiYouJi-6b-AWQ)  |
| 水浒传风格 | 🤗 [stylellm/ShuiHuZhuan-6b-AWQ](https://huggingface.co/stylellm/ShuiHuZhuan-6b-AWQ)  |
| 红楼梦风格 | 🤗 [stylellm/HongLouMeng-6b-AWQ](https://huggingface.co/stylellm/HongLouMeng-6b-AWQ)  |
> **注意：**<br/>
> 使用AWQ量化模型前需安装[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)：pip install autoawq <br/>
> 量化后的模型占用显存4.2GB，适用于较小的显卡使用，但风格表达能力有所下降，追求完整效果的场景请使用[量化前版本](#模型列表)

# 风格转化效果
以《白雪公主》原文作为输入，以下是不同风格模型的改写结果。

| 原文 - 《白雪公主》     | 改写-三国演义风格 | 改写-西游记风格 | 改写-水浒传风格 | 改写-红楼梦风格 |
|:----------|:----------|:----------|:----------|:----------|
| 严冬时节，鹅毛一样的大雪片在天空中到处飞舞着，有一个王后坐在王宫里的一扇窗子边，正在为她的女儿做针线活儿，寒风卷着雪片飘进了窗子，乌木窗台上飘落了不少雪花。 | 时值隆冬，大雪压境，王后坐于王宫之窗下，正与女作绣，寒风入室，吹落乌木窗台之雪。 | 正是那：朔风吹凛凛，瑞雪霭纷纷；王后坐窗前，正教女作绣，只见那：寒风送雪入窗棂，乌木台头堆积雪。 | 此时正是隆冬之际，鹅毛般大雪乱飘，有一王后坐于王宫内一扇窗户前，正缝其女衣，寒风卷雪入得窗来，把那乌木窗台都压坏了。 | 且说那严冬天气，鹅毛大雪，纷纷扬扬，只见那王后坐于王宫之窗下，正教他女儿作针黹，忽见寒风卷雪，吹入窗内，乌木窗台之上，积了半日之雪。 |
| 她抬头向窗外望去，一不留神，针刺进了她的手指，红红的鲜血从针口流了出来，有三点血滴落在飘进窗子的雪花上。 | 其女仰视窗外，不料针落指中，遂出血于雪，有三点之痕。 | 只见那女子举目望外观看时，不期把个指头儿扎了一针，又见出血，却似三两点血珠儿落于雪内。 | 那娘子仰面看时，只道是雪片落将下来，不提防把针儿捻在手里，一时失手，正中指尖，便见出血来，有三两点，都滴入窗里去。 | 只见那丫头往外瞧时，一时针扎手了，便出血了，有三四个血点儿，落在了雪里。 |
| 她若有所思地凝视着点缀在白雪上的鲜红血滴，又看了看乌木窗台，说道："但愿我小女儿的皮肤长得白里透红，看起来就像这洁白的雪和鲜红的血一样，那么艳丽，那么骄嫩，头发长得就像这窗子的乌木一般又黑又亮！" | 其女乃凝眸望那点缀于白雪之下的鲜红血滴，复观乌木窗棂曰：“吾愿吾小女肌肤如雪，容貌似此洁白之雪与鲜红之血，娇嫩之发，正如此窗棂之乌木也。” | 只见那般血迹，却也殷红；又见那般白雪，却也莹洁。她又望了望那乌木窗棂道：“但愿吾女肌肤如雪，颜色似朱，发髻如漆，方才好哩！” | 只见那女子把眼来望那白雪上点点鲜血，又看那乌木窗子道："只愿俺的小女儿皮肉也似这般白净，便如这雪相似；须发也似这般漆黑光鲜，便似这乌木般好！" | 一面看那白雪中点点血痕，一面又瞧那乌木窗棂，因叹道：“只望我小女肌肤丰泽，如这雪白，血红，发也似的乌木一般。” |
| 她的小女儿渐渐长大了，小姑娘长得水灵灵的，真是人见人爱，美丽动人。她的皮肤真的就像雪一样的白嫩，又透着血一样的红润，头发像乌木一样的黑亮。所以王后给她取了个名字，叫白雪公主。 | 其女渐长，乃名曰：“白雪”，其色如雪，真美之极。肤若冰雪，发似乌木，故王后以白雪为号。 | 光阴似箭，不觉数载，那小女儿渐长成人，生得娉婷娇艳，真个是：肤如凝脂，貌若桃李，发似漆黑，故此王后名曰白雪公主。 | 那小女儿渐长成，生得十分美貌，真个有倾国之色，娇艳无伦。肌肤如玉，面似桃花，发若漆黑。因此王后便取名唤做"白雪公主"。 | 那丫头渐长，生得袅娜纤巧，乖觉伶俐，肌肤丰泽，容貌娇美，且是雪肌花颜，唇若施脂，腰如束柳。故此，王后便命名曰：“白雪”。 |
| 但白雪公主还没有长大，她的王后妈妈就死去了。不久，国王爸爸又娶了一个妻子。这个王后长得非常漂亮，但她很骄傲自负，嫉妒心极强，只要听说有人比她漂亮，她都不能忍受。 | 却说白雪公主年方幼小，其母王后乃亡矣。不数日，父王复纳新妇，此妇美而骄妒，闻人言有胜己者，不能容也。 | 那白雪公主尚幼年，其母王后已亡矣。不多时，国王复纳一妻，此女貌美而骄矜，妒忌之心甚重，闻得人言有胜于己者，即不能容也。 | 此时白雪公主尚幼小，其母王后便亡故了；不数日间，父王复纳新妇为妻。此女美貌绝伦，性傲气高，妒忌之心甚重，闻有胜己者，便不能容。 | 且说那白雪公主尚未长成，其母王后已亡故了。不几日，国王复纳一妻，此妃生得甚美，惟性傲慢，妒忌之心颇重，凡见人胜过己者，便不能容。 |
| 她有一块魔镜，她经常走到镜子面前自我欣赏，并问道："告诉我，镜子，告诉我实话！这儿所有的女人谁最漂亮？告诉我她是谁？"镜子回答道："是你，王后！你就是这儿最漂亮的女人。"听到这样的话，她就会满意地笑起来。 | 王后乃有魔镜，常自照其面曰：“汝言我，镜中实也！此间诸女何者最美？吾告汝其人。”镜答曰：“汝也，王后也！汝乃此处之绝色。”闻之，遂喜。 | 她又有个魔镜，常来照见，便问：“镜子呀，你且说个明白！这众女子中那一个最美？快教我认得他哩！”那镜子即答曰：“娘娘乃天下第一美人也。”闻言，她就欢喜而笑。 | 那娘子有面魔镜，常自照影，便问："你且说与我，镜中甚人最美貌？是何女子？"镜答曰："娘子乃此间第一美人也。"听得这话，便欢喜。 | 她又有一个宝镜，每日临镜自照，或问于镜曰：“我何人哉？”那镜便答道：“你乃天下第一美人儿！”如此一回，她就喜之不尽了。 |
| 但白雪公主慢慢地长大，并出落得越来越标致漂亮了。到了七岁时，她长得比明媚的春光还要艳丽夺目，比王后更美丽动人。直到有一天，王后像往常一样地去问那面魔镜时，镜子作出了这样的回答："王后，你是美丽漂亮的，但是白雪公主要比你更加漂亮！"她听到了这话，心里充满了愤怒和妒忌，脸也变得苍白起来。 | 然则白雪公主渐渐长成，其美乃胜于昔日，至七岁之时，其容貌又胜王后。一日，王后如常问镜曰：“吾乃绝世之色，今白雪公主乃胜我矣！”闻言大怒，面亦变青。 | 光阴似箭，不觉又过了七年，白雪公主渐渐长成，却生得如花似玉，比那妩媚之春，胜似娇娘；及至王后来照那面魔镜，只见那镜中答道：“王后啊，你虽美貌，怎比得上白雪公主？”她闻言大怒，顿觉愁容惨淡，颜色憔悴。 | 且说白雪公主渐渐长成，生得十分美貌，至七岁之时，胜似春光，赛过王后。一日，王后照旧去问那面魔镜，镜中答道："王后，你虽是极妙绝伦，却怎比得上白雪公主？"听得此言，心中忿怒，面如纸色。 | 且说白雪公主渐渐长成，生得不俗，至七岁上，便胜似春光，又胜过王后。一日，王后来照魔镜，镜中答道：“王后啊！你虽美貌，岂及白雪公主？”闻此言，心内大怒，容颜亦变。 |
| 她叫来了一名仆人对他说："给我把白雪公主抓到大森林里去，我再也不希望看到她了。"仆人把白雪公主带走了。在森林里他正要动手杀死她时，她哭泣着哀求他不要杀害她。 | 其女唤一婢曰：“速将白雪公主擒至大林中，吾不复见之。”婢遂引白雪公主而去。行至林内，欲杀之，公主泣告勿杀。 | 遂唤一婢，教：“快与我拿白雪公主于大林中，再不许见也！”那婢即引白雪公主而去。及至林中，欲杀之，只见她哭告曰：“乞勿伤我性命。” | 那娘子唤得一个使女，分付道："且将白雪公主捉到大林子里去，休教见我。"便将白雪公主掣出林外，欲杀之，哭告勿行。 | 只见那丫头便唤了一个使女来说道：“你且将白雪公主捉来，我那里还见得？”那使女领命去了，至大林中欲杀之，哭诉不饶。 |
| 面对楚楚动人的可怜小公主的哀求，仆人的同情之心油然而生，他说道："你是一个人见人爱的孩子，我不会杀害你。"这样，他把她单独留在了森林里。 | 其仆怜楚楚之幼女，曰：“汝乃可爱之人，吾不杀汝。”遂将孤置于林中。 | 那小公主又哭得凄惨，那仆人心下甚怜，遂言曰：“汝乃好身儿，我岂杀汝？”遂将她独自留在林中。 | 那可怜的小公主楚楚动人，哭得仆人心软，便道："汝乃天姿国色，吾岂忍杀？且将汝独自林中寄放。" | 那小丫头儿哭得可怜，又见仆人怜惜，便说道：“你是好人儿，我岂肯害你？”遂将她一人留在林中。 |
| 当仆人决定不再杀害白雪公主，而把她留在那儿时，尽管他知道在那荒无人际的大森林里，她十有八九会被野兽撕成碎片，但想到他不必亲手杀害她，他就觉得压在心上的一块沉重的大石头落了下来。 | 其仆见不杀白雪公主，留于彼处，虽知大林之中，必被虎狼食尽，然念吾不用亲刃之，心下遂宽了一块重石。 | 只见那厮家奴不肯杀白雪公主，却将她留此荒林之中，虽知必遭禽兽之嚼，然不须亲下手也，遂宽心矣。 | 仆人自思：若要杀白雪公主，便在此处；又恐此间大林子里，被禽兽啖食了，不如且留她在那里，我亦无须亲自下手。遂放了她去。 | 及至仆人主意已定，不欲加害于白雪公主，遂留之不去，虽知大林深处，必遭禽兽吞噬，然想自己无须亲自动手，便如卸了千钧重担一般。 |
|  | **🔍[查看全文](https://github.com/stylellm/stylellm_models/blob/main/examples/SanGuoYanYi-6b/output_example.md)** | **🔍[查看全文](https://github.com/stylellm/stylellm_models/blob/main/examples/XiYouJi-6b/output_example.md)** | **🔍[查看全文](https://github.com/stylellm/stylellm_models/blob/main/examples/ShuiHuZhuan-6b/output_example.md)** | **🔍[查看全文](https://github.com/stylellm/stylellm_models/blob/main/examples/HongLouMeng-6b/output_example.md)** |
<br />
  
    

# 风格化聊天效果
利用StyleLLM模型对ChatGPT等通用大模型的输出进行风格改写，可改变对话风格单一、AI味过重的状况。（详细介绍见
[StyleLLM-Chat](https://github.com/stylellm/stylellm_chat)）

### "水浒传-鲁智深"对话风格效果：
![demo1](https://github.com/stylellm/stylellm_chat/blob/main/%20assets/gradio_chat_1.png)
![demo2](https://github.com/stylellm/stylellm_chat/blob/main/%20assets/gradio_chat_2.png)

### 在Colab中快速对话：
[打开Colab](https://colab.research.google.com/github/stylellm/stylellm_chat/blob/main/colab_web_demo_gradio.ipynb) 


# 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stylellm/SanGuoYanYi-6b")
model = AutoModelForCausalLM.from_pretrained("stylellm/SanGuoYanYi-6b").eval()

messages = [{"role": "user", "content": "严冬时节，鹅毛一样的大雪片在天空中到处飞舞着，有一个王后坐在王宫里的一扇窗子边，正在为她的女儿做针线活儿，寒风卷着雪片飘进了窗子，乌木窗台上飘落了不少雪花。"}]
input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print("Output:", response)
# Output: 时值隆冬，大雪压境，王后坐于王宫之窗下，正与女作绣，寒风入室，吹落乌木窗台之雪。
```


# 致谢
本项目在模型训练中借助了下列开源项目，在此对相关项目人员表示诚挚感谢：
- [01-ai/Yi](https://github.com/01-ai/Yi):  本项目模型底座
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):  本项目训练框架
