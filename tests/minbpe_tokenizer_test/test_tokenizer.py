import sys

from minbpe_tokenizer import data

print(sys.path)
from minbpe_tokenizer.tokenizer import BasicTokenizer, RegexTokenizer

TEXT = (
    "The quick brown fox jumps over the lazy dog. It's amazing how many languages and symbols "
    "exist in the world today! 🚀 When you look at the complexity of communication, you realize how "
    "interconnected our global society is. 🌏 Did you know that in Mandarin Chinese, '你好' (nǐ hǎo) "
    "means 'hello'? It's one of the most widely spoken languages. And in Greek, 'Γειά σας' (Yia sas) "
    "is used for greetings! In Arabic, 'مرحبًا' (Marhaban) is a warm way to say hi to someone new. "
    "Each language has its own beauty, rich history, and unique expressions. 😊\n\n"
    "In some cultures, greetings might include a handshake or a hug, and in others, a bow. In Japan, "
    "for example, they use 'こんにちは' (Konnichiwa), which literally translates to 'Good day.' Meanwhile, "
    "in Russian, 'Привет' (Privet) is a casual greeting among friends. 🇷🇺\n\n"
    "Symbols like '❤️' can express love universally, no matter the language. The use of emoji, such as "
    "'🙂', has become an essential part of digital communication. Whether you're writing a text message or "
    "an email, it adds a layer of emotion and personality to our words. 📝\n\n"
    "It's also fascinating how mathematics and science use symbols that transcend language barriers. For "
    "example, 'π' represents the mathematical constant, and 'Σ' is the symbol for summation in mathematics. "
    "Equations like E = mc² demonstrate how even science has its own universal language. 🌌\n\n"
    "In the world of art and design, symbols like '⚡' or '★' can convey power or excellence. Similarly, "
    "'∞' represents infinity, suggesting the limitless possibilities within our world. From ancient Egyptian "
    "hieroglyphs to modern digital icons, symbols have been a vital part of human expression. 🖋\n\n"
    "But it’s not just about words and symbols—our entire digital world is built on code. Binary numbers, such "
    "as '01001001', represent everything we see on a screen. And even deeper, the UTF-8 Unicode encoding system "
    "allows us to write in different languages and scripts, from Latin to Arabic to Devanagari and beyond! The "
    "richness of human language and communication continues to evolve. 🌍\n\n"
    "By understanding and embracing the variety of languages, symbols, and scripts that exist around the world, "
    "we can better appreciate the diversity and complexity of human culture. ✨\n\n"
    "Thank you for reading this message, and remember: '世界是美丽的' (Shìjiè shì měilì de), meaning "
    "'The world is beautiful' in Mandarin. 😊🙏"
)


def test_tokenizer_encode():
    tokenizer = BasicTokenizer()
    tokenizer._vocab[256] = (32, 116) # space t
    tokenizer._vocab[257] = (32, 115) # space s
    encoded = tokenizer.encode(text=TEXT)
    assert len(encoded) < len(TEXT.encode("utf-8"))

def test_tokenizer_decode():
    tokenizer = BasicTokenizer()
    tokenizer._vocab[256] = (32, 116) # space t
    tokenizer._vocab[257] = (32, 115) # space s
    encoded = tokenizer.encode(text=TEXT)
    encoded_decoded_text = tokenizer.decode(encoded)
    assert TEXT == encoded_decoded_text

def test_basic_tokenizer_train():
    tokenizer = BasicTokenizer()
    # tokenizer.train(data.training_text, verbose=True)
    tokenizer.train(data.training_text, verbose=True)
    encoded = tokenizer.encode(text=TEXT)
    print(f"Len of TEXT: {len(TEXT.encode('utf-8'))}. Len of encoded TEXT: {len(encoded)}")
    assert len(encoded) < len(TEXT.encode("utf-8"))
    encoded_decoded_text = tokenizer.decode(encoded)
    assert TEXT == encoded_decoded_text

def test_tokenizer_save(tmp_path):
    file_path = tmp_path.joinpath("vocab.json")
    tokenizer1 = BasicTokenizer()
    tokenizer1.train(TEXT, verbose=False, vocab_size=275)
    encoded1 = tokenizer1.encode(TEXT)
    tokenizer1.save(file_path)
    tokenizer2 = BasicTokenizer.from_file(file_path)
    encoded2 = tokenizer2.encode(TEXT)
    assert encoded1 == encoded2
    assert TEXT == tokenizer2.decode(encoded2)

def test_regex_tokenizer_train():
    tokenizer = RegexTokenizer()
    tokenizer.train(data.training_text, vocab_size=512, verbose=True)
    encoded = tokenizer.encode(text=TEXT)
    print(f"Len of TEXT: {len(TEXT.encode('utf-8'))}. Len of encoded TEXT: {len(encoded)}")
    assert len(encoded) < len(TEXT.encode("utf-8"))
    encoded_decoded_text = tokenizer.decode(encoded)
    assert TEXT == encoded_decoded_text

