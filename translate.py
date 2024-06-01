import tkinter as tk
from tkinter import scrolledtext
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "chinese": "zh",
    "japanese": "jap",
    "korean": "ko",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "russian": "ru",
    "ukrainian": "uk"
}

def get_language_code(language_name):
    return LANGUAGE_CODES.get(language_name.lower().strip(), None)

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Translate.py")
        self.languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Dutch", "Russian", "Ukrainian"]

        self.src_lang_label = tk.Label(root, text="Source language:")
        self.src_lang_label.pack()
        self.src_lang_var = tk.StringVar(root)
        self.src_lang_var.set(self.languages[0])
        self.src_lang_menu = tk.OptionMenu(root, self.src_lang_var, *self.languages)
        self.src_lang_menu.pack()

        self.tgt_lang_label = tk.Label(root, text="Target language:")
        self.tgt_lang_label.pack()
        self.tgt_lang_var = tk.StringVar(root)
        self.tgt_lang_var.set(self.languages[1])
        self.tgt_lang_menu = tk.OptionMenu(root, self.tgt_lang_var, *self.languages)
        self.tgt_lang_menu.pack()

        self.model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(get_language_code(self.languages[0]), get_language_code(self.languages[1]))
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

        self.input_label = tk.Label(root, text="Input text:")
        self.input_label.pack()
        self.input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
        self.input_box.pack(padx=10, pady=10)

        self.process_button = tk.Button(root, text="Process text", command=self.process_text)
        self.process_button.pack(pady=10)

        self.output_label = tk.Label(root, text="Processed text:")
        self.output_label.pack()
        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10, state=tk.DISABLED)
        self.output_box.pack(padx=10, pady=10)

    def process_text(self):
        input_text = self.input_box.get("1.0", tk.END).strip()
        src_lang_name = self.src_lang_var.get()
        target_lang_name = self.tgt_lang_var.get()

        src_lang = get_language_code(src_lang_name)
        target_lang = get_language_code(target_lang_name)
        model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src_lang, target_lang)
        if model_name != self.model_name:
            self.model_name = model_name
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)

        processed_text = self.translate_text(input_text, src_lang, target_lang)
        
        self.output_box.config(state=tk.NORMAL)
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.INSERT, processed_text)
        self.output_box.config(state=tk.DISABLED)

    def translate_text(self, text, src_lang, target_lang):
        target_token = f">>{target_lang}<<"
        sentences = ["{} {}".format(target_token, sentence) for sentence in sent_tokenize(text)]
        max_len = 512
        inputs = self.tokenizer(sentences, return_tensors="pt", max_length=max_len, padding=True, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=max_len, num_beams=4, early_stopping=True)
        decoded_sentences = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        decoded_text = " ".join(decoded_sentences)
        
        return decoded_text


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)

    root.mainloop()
